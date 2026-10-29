//! A key and its value in one block: what a bucket of a [`super::ShardMap`]
//! points to.
//!
//! As a `(Key, KiviDBValue)` in the table, every bucket was forty bytes, and
//! a table sits between half and seven eighths full: a key of a 100-byte
//! string cost the table about 86 bytes before its value's own block. Here a
//! bucket is the eight bytes of a [`Slot`], and the key, a header and (for a
//! string that fits) the value's bytes share one block, from the slabs when
//! it is small enough (`value_slab`, up to 4 KiB).
//!
//! **Layout.**
//!
//! ```text
//! 0   u32  holders (low 24 bits) | form (top 8)
//! 4   u32  reserved for use times (LRU/LFU), zero for now
//! 8   u8   key length, or LONG_KEY: the length is the u32 at 12
//! 9   u8   flags
//! 10  u16  length of an inline string
//! 12  [u32 key length]   only for a long key
//!     key bytes
//!     an inline string's bytes, or, 8-aligned, a KiviDBValue kept whole
//! ```
//!
//! A string short enough is kept inline: one block for key and value, and
//! nothing else to visit. Anything else (a collection, or a string too long
//! to be inline) is its `KiviDBValue`, kept whole in the block: its handle
//! is borrowed in place, so a collection is reached as before.
//!
//! **Holders.** The table holds each entry once. A frozen snapshot, and
//! whatever else must read a key and its value once the shard is unlocked,
//! takes another hold ([`EntryRef`]); the block goes when the last hold
//! does. Nothing in a block is written after it is made, but by compaction
//! moving the block whole, which it does only when the table's is the only
//! hold, and only with the shard locked for writing.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering, fence};

use super::shard_map::{StrRef, ValueRef};
use crate::shared_bytes::SharedBytes;
use crate::types::KiviDBValue;
use crate::value_slab;

#[repr(C)]
struct Header {
    word: AtomicU32,
    lru: AtomicU32,
    klen: u8,
    flags: u8,
    vlen: u16,
}

const HEADER: usize = std::mem::size_of::<Header>();
const _: () = assert!(HEADER == 12);

const FORM_SHIFT: u32 = 24;
const HOLDERS: u32 = (1 << FORM_SHIFT) - 1;
/// More holds than this and the process stops, as `Arc` does long before
/// its count could wrap into the form.
const MAX_HOLDERS: u32 = 1 << 23;

/// A string whose bytes follow the key.
const INLINE: u32 = 1;
/// A `KiviDBValue` kept whole after the key.
const WHOLE: u32 = 2;

/// `klen` of a key of 255 bytes or more: its length is the u32 after the
/// header.
const LONG_KEY: u8 = u8::MAX;

/// Blocks are aligned to this; slab blocks are, and the system allocator's
/// are asked to be.
const ALIGN: usize = 16;

/// The longest entry a string's bytes are kept inline in: the largest block
/// a slab holds.
pub const INLINE_MAX: usize = value_slab::MAX_BLOCK;

/// Where the key starts, for a key of `klen` bytes. Without a branch: it
/// is asked for on every lookup.
#[inline(always)]
const fn key_at(klen: usize) -> usize {
    HEADER + 4 * (klen >= LONG_KEY as usize) as usize
}

/// Where a whole `KiviDBValue` goes, after a key of `klen` bytes.
#[inline]
const fn whole_at(klen: usize) -> usize {
    (key_at(klen) + klen).next_multiple_of(8)
}

#[inline]
fn inline_size(klen: usize, vlen: usize) -> usize {
    key_at(klen) + klen + vlen
}

#[inline]
fn whole_size(klen: usize) -> usize {
    whole_at(klen) + std::mem::size_of::<KiviDBValue>()
}

/// Whether a string of `vlen` bytes under a key of `klen` is kept inline.
#[inline]
fn fits_inline(klen: usize, vlen: usize) -> bool {
    vlen <= u16::MAX as usize && inline_size(klen, vlen) <= INLINE_MAX
}

fn allocate(size: usize) -> NonNull<u8> {
    match value_slab::class_of(size) {
        Some(class) => value_slab::alloc(class),
        None => {
            let layout = Layout::from_size_align(size, ALIGN).expect("an entry's layout");
            // SAFETY: the size is never zero: there is a header.
            match NonNull::new(unsafe { std::alloc::alloc(layout) }) {
                Some(at) => at,
                None => std::alloc::handle_alloc_error(layout),
            }
        }
    }
}

/// # Safety
/// `at` came from [`allocate`] of `size`, and is not used again.
unsafe fn deallocate(at: NonNull<u8>, size: usize) {
    match value_slab::class_of(size) {
        // SAFETY: as the caller promises.
        Some(class) => unsafe { value_slab::free(at, class) },
        None => unsafe {
            std::alloc::dealloc(
                at.as_ptr(),
                Layout::from_size_align(size, ALIGN).expect("an entry's layout"),
            )
        },
    }
}

/// An entry's address: what a slot holds. Copying it takes no hold; whoever
/// uses one has to know the entry lives (a shard lock, or a hold).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Entry(NonNull<Header>);

// SAFETY: an entry is shared only as described above, and what it holds
// (bytes, `KiviDBValue`) is `Send` and `Sync`.
unsafe impl Send for Entry {}
unsafe impl Sync for Entry {}

impl Entry {
    /// A new entry, with one hold: the caller's (the table's, once in it).
    pub(super) fn new(key: &[u8], value: KiviDBValue) -> Entry {
        if let KiviDBValue::String(bytes) = &value
            && fits_inline(key.len(), bytes.len())
        {
            return Self::inline(key, bytes);
        }
        let klen = key.len();
        let at = Self::begin(key, whole_size(klen), WHOLE, 0);
        // SAFETY: the block has room for the value at `whole_at`, aligned
        // for it, and nothing else has the block yet.
        unsafe {
            (at.as_ptr() as *mut u8)
                .add(whole_at(klen))
                .cast::<KiviDBValue>()
                .write(value);
        }
        Entry(at)
    }

    /// A new entry for the string `bytes`: inline if it fits, else in a
    /// `SharedBytes` of its own.
    pub(super) fn string(key: &[u8], bytes: &[u8]) -> Entry {
        if fits_inline(key.len(), bytes.len()) {
            Self::inline(key, bytes)
        } else {
            Self::new(key, KiviDBValue::String(SharedBytes::from(bytes)))
        }
    }

    fn inline(key: &[u8], bytes: &[u8]) -> Entry {
        let klen = key.len();
        let at = Self::begin(key, inline_size(klen, bytes.len()), INLINE, bytes.len());
        // SAFETY: the block has room for the bytes after the key.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (at.as_ptr() as *mut u8).add(key_at(klen) + klen),
                bytes.len(),
            );
        }
        Entry(at)
    }

    /// A block of `size`, its header and key written.
    fn begin(key: &[u8], size: usize, form: u32, vlen: usize) -> NonNull<Header> {
        let klen = key.len();
        assert!(klen <= u32::MAX as usize, "a key of {klen} bytes");
        let at = allocate(size).cast::<Header>();
        // SAFETY: `size` covers the header, the long key's length and the
        // key, and the block is 16-aligned and nobody else's.
        unsafe {
            at.as_ptr().write(Header {
                word: AtomicU32::new(1 | form << FORM_SHIFT),
                lru: AtomicU32::new(0),
                klen: klen.min(LONG_KEY as usize) as u8,
                flags: 0,
                vlen: vlen as u16,
            });
            let base = at.as_ptr() as *mut u8;
            if klen >= LONG_KEY as usize {
                base.add(HEADER).cast::<u32>().write_unaligned(klen as u32);
            }
            std::ptr::copy_nonoverlapping(key.as_ptr(), base.add(key_at(klen)), klen);
        }
        at
    }

    #[inline]
    fn header(&self) -> &Header {
        // SAFETY: the entry lives (see the type).
        unsafe { self.0.as_ref() }
    }

    #[inline]
    pub(super) fn base(self) -> *const u8 {
        self.0.as_ptr() as *const u8
    }

    #[inline]
    fn form(&self) -> u32 {
        self.header().word.load(Ordering::Relaxed) >> FORM_SHIFT
    }

    #[inline(always)]
    fn klen(&self) -> usize {
        let short = self.header().klen;
        if short == LONG_KEY {
            // SAFETY: a long key's length follows the header.
            unsafe { self.base().add(HEADER).cast::<u32>().read_unaligned() as usize }
        } else {
            short as usize
        }
    }

    /// Whether the entry's key is `key`. A key of 8 to 16 bytes (most keys
    /// are) is compared as two words, without a call to `memcmp`.
    #[inline(always)]
    pub(super) fn has_key(self, key: &[u8]) -> bool {
        let mine = self.key();
        let n = key.len();
        if mine.len() != n {
            return false;
        }
        if (8..=16).contains(&n) {
            let word = |s: &[u8], at: usize| u64::from_ne_bytes(s[at..at + 8].try_into().unwrap());
            return word(mine, 0) == word(key, 0) && word(mine, n - 8) == word(key, n - 8);
        }
        mine == key
    }

    /// The key's bytes, for as long as the entry lives.
    #[inline(always)]
    pub(super) fn key<'a>(self) -> &'a [u8] {
        let klen = self.klen();
        // SAFETY: `klen` bytes of key at `key_at`, never written again.
        unsafe { std::slice::from_raw_parts(self.base().add(key_at(klen)), klen) }
    }

    /// The key, as the `&str` it was made from.
    #[inline]
    pub(super) fn key_str<'a>(self) -> &'a str {
        // SAFETY: every key is made from a `&str` (`ShardMap::insert` and
        // friends take one).
        unsafe { std::str::from_utf8_unchecked(self.key()) }
    }

    /// The value, for as long as the entry lives.
    #[inline(always)]
    pub(super) fn value<'a>(self) -> ValueRef<'a> {
        let klen = self.klen();
        if self.form() == INLINE {
            let vlen = self.header().vlen as usize;
            // SAFETY: `vlen` bytes follow the key, never written again.
            let bytes =
                unsafe { std::slice::from_raw_parts(self.base().add(key_at(klen) + klen), vlen) };
            ValueRef::String(StrRef::bytes(bytes))
        } else {
            ValueRef::from(self.whole())
        }
    }

    /// The `KiviDBValue` of a [`WHOLE`] entry.
    #[inline]
    fn whole<'a>(self) -> &'a KiviDBValue {
        debug_assert_eq!(self.form(), WHOLE);
        // SAFETY: a whole entry's value is at `whole_at`, aligned, and
        // lives as long as the entry.
        unsafe { &*self.base().add(whole_at(self.klen())).cast::<KiviDBValue>() }
    }

    /// The block's size.
    #[inline]
    pub(super) fn size(self) -> usize {
        let klen = self.klen();
        if self.form() == INLINE {
            inline_size(klen, self.header().vlen as usize)
        } else {
            whole_size(klen)
        }
    }

    #[inline]
    fn holders(self) -> u32 {
        self.header().word.load(Ordering::Acquire) & HOLDERS
    }

    /// One more hold. The entry must live: the caller holds the shard, or a
    /// hold of its own.
    #[inline]
    pub(super) fn retain(self) {
        let before = self.header().word.fetch_add(1, Ordering::Relaxed) & HOLDERS;
        if before >= MAX_HOLDERS {
            std::process::abort();
        }
    }

    /// Let go of one hold; the last frees the entry and its value.
    ///
    /// # Safety
    /// The caller has a hold, and does not use the entry again.
    pub(super) unsafe fn release(self) {
        if self.header().word.fetch_sub(1, Ordering::Release) & HOLDERS != 1 {
            return;
        }
        // What the other holders did with the entry comes before it goes.
        fence(Ordering::Acquire);
        let size = self.size();
        if self.form() == WHOLE {
            // SAFETY: the last hold: nothing reads the value any more.
            unsafe {
                std::ptr::drop_in_place(self.base().add(whole_at(self.klen())) as *mut KiviDBValue)
            };
        }
        // SAFETY: the block came from `allocate(size)`.
        unsafe { deallocate(self.0.cast(), size) };
    }

    /// The value, owned, and the caller's hold given up. A string kept
    /// inline is copied out; anything else is another handle on it.
    ///
    /// # Safety
    /// The caller has a hold, and does not use the entry again.
    pub(super) unsafe fn into_value(self) -> KiviDBValue {
        let value = self.value().to_value();
        // SAFETY: as the caller promises.
        unsafe { self.release() };
        value
    }

    /// Move the entry, or its string's bytes, out of a slab compaction is
    /// emptying (`regions` is `value_slab::regions_snapshot`), if the table's
    /// is its only hold. Returns the new entry to put in its slot, if the
    /// entry moved, and what was left behind, to free with the shard
    /// unlocked.
    ///
    /// # Safety
    /// The caller holds the shard for writing, and the table holds the entry.
    pub(super) unsafe fn relocate(self, regions: &[usize]) -> Relocation {
        let size = self.size();
        let entry_moves =
            value_slab::class_of(size).is_some() && value_slab::is_evacuating(self.base(), regions);
        let bytes_move = self.form() == WHOLE
            && matches!(self.whole(), KiviDBValue::String(b)
                if value_slab::is_evacuating(b.block(), regions));
        if !entry_moves && !bytes_move {
            return Relocation::Stays;
        }
        if self.holders() != 1 {
            return Relocation::Held;
        }
        if entry_moves {
            let to = allocate(size);
            // SAFETY: the only hold is the table's, and the shard is held
            // for writing: nothing reads the block while it is copied, and
            // nothing but the slot knows where it is. The copy is the
            // value's move; the old block is freed without dropping it.
            unsafe { std::ptr::copy_nonoverlapping(self.base(), to.as_ptr(), size) };
            return Relocation::Moved {
                to: Entry(to.cast()),
                left: Left(self.0.cast(), size),
            };
        }
        // SAFETY: as above, nothing else reads the value.
        let value = unsafe { &mut *(self.base().add(whole_at(self.klen())) as *mut KiviDBValue) };
        match value {
            KiviDBValue::String(bytes) => match bytes.relocate(regions) {
                Some(old) => Relocation::BytesMoved(old),
                None => Relocation::Held,
            },
            _ => Relocation::Stays,
        }
    }
}

/// What [`Entry::relocate`] did.
pub(super) enum Relocation {
    Stays,
    Held,
    /// The entry moved to `to`; `left` is the old block.
    Moved {
        to: Entry,
        left: Left,
    },
    /// The entry stayed; its string's bytes moved, and this is the old block.
    BytesMoved(SharedBytes),
}

/// An entry's block left behind by compaction, freed when dropped.
pub struct Left(NonNull<u8>, usize);

// SAFETY: nothing but this knows the block.
unsafe impl Send for Left {}

impl Drop for Left {
    fn drop(&mut self) {
        // SAFETY: the block was an entry's, copied elsewhere; its value
        // moved with the copy.
        unsafe { deallocate(self.0, self.1) };
    }
}

/// A hold on an entry: its key and value can be read with no shard locked,
/// and neither changes while it is held. What a frozen snapshot keeps of a
/// string.
pub struct EntryRef(Entry);

impl EntryRef {
    /// # Safety
    /// The entry lives: the caller holds its shard.
    pub(super) unsafe fn new(entry: Entry) -> Self {
        entry.retain();
        EntryRef(entry)
    }

    pub fn key(&self) -> &str {
        self.0.key_str()
    }

    pub fn value(&self) -> ValueRef<'_> {
        self.0.value()
    }

    /// Whether `entry` is this one.
    pub(super) fn is(&self, entry: Entry) -> bool {
        self.0 == entry
    }
}

impl Clone for EntryRef {
    fn clone(&self) -> Self {
        self.0.retain();
        EntryRef(self.0)
    }
}

impl Drop for EntryRef {
    fn drop(&mut self) {
        // SAFETY: this is a hold.
        unsafe { self.0.release() };
    }
}

/// An entry the table has let go of: a key's value as it was replaced or
/// removed. Its value can be read, or taken; it is freed when dropped (with
/// the shard unlocked, ideally: freeing is the allocator's time).
pub struct Taken(Entry);

impl Taken {
    /// # Safety
    /// The caller gives this the table's hold.
    pub(super) unsafe fn new(entry: Entry) -> Self {
        Taken(entry)
    }

    pub fn key(&self) -> &str {
        self.0.key_str()
    }

    pub fn value(&self) -> ValueRef<'_> {
        self.0.value()
    }

    /// The value, owned.
    pub fn into_value(self) -> KiviDBValue {
        let entry = self.0;
        std::mem::forget(self);
        // SAFETY: the hold was this one's.
        unsafe { entry.into_value() }
    }
}

impl Drop for Taken {
    fn drop(&mut self) {
        // SAFETY: the hold is this one's.
        unsafe { self.0.release() };
    }
}

/// What a bucket holds: the entry's address in the low 48 bits, and in the
/// top 16 a 12-bit fragment of the key's hash and a 4-bit hint of the
/// entry's size.
///
/// The fragment is compared before the entry is touched. hashbrown's 7-bit
/// tag lets one probed key in 128 through by chance, and each of those
/// was a visit to an entry somewhere in the whole dataset; with the fragment,
/// one in half a million is. When the fragment matches, the entry's first
/// lines are asked for at once, the hint saying how many: a large value's
/// copy otherwise waits for the header's line before the rest is asked for.
#[derive(Clone, Copy)]
pub(super) struct Slot(u64);

const ADDRESS_BITS: u32 = 48;
const ADDRESS: u64 = (1 << ADDRESS_BITS) - 1;
/// Cache lines for each size hint.
const HINT_LINES: [u8; 16] = [1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64];
/// The most lines asked for ahead: past these the hardware's own prefetcher
/// has seen the stream.
pub(super) const PREFETCH_LINES: usize = 8;

impl Slot {
    #[inline]
    fn fragment(hash: u64) -> u64 {
        // Bits hashbrown uses for neither the bucket (the low bits) nor its
        // tag (the top seven).
        (hash >> 32) & 0xfff
    }

    pub(super) fn new(entry: Entry, hash: u64) -> Slot {
        // The address is exposed: the slot holds it as a number, and
        // `entry` turns it back into a pointer to the same block.
        let at = entry.0.as_ptr().expose_provenance() as u64;
        // Every address user space is given without asking for a high one
        // is below 2^47 (x86-64) or 2^48 (arm64), on Linux, Windows and
        // macOS alike.
        assert_eq!(at & !ADDRESS, 0, "an entry above 2^48");
        let lines = entry.size().div_ceil(64);
        let hint = HINT_LINES
            .iter()
            .position(|&l| l as usize >= lines)
            .unwrap_or(15) as u64;
        Slot(at | Self::fragment(hash) << 52 | hint << ADDRESS_BITS)
    }

    /// The same slot for the entry moved to `to` (by compaction): the
    /// fragment and the hint stay.
    #[inline]
    pub(super) fn moved_to(self, to: Entry) -> Slot {
        let at = to.0.as_ptr().expose_provenance() as u64;
        assert_eq!(at & !ADDRESS, 0, "an entry above 2^48");
        Slot(self.0 & !ADDRESS | at)
    }

    #[inline(always)]
    pub(super) fn entry(self) -> Entry {
        // SAFETY: made from an entry's address, never zero.
        Entry(unsafe {
            NonNull::new_unchecked(std::ptr::with_exposed_provenance_mut::<Header>(
                (self.0 & ADDRESS) as usize,
            ))
        })
    }

    /// Whether this can be the entry of a key whose hash is `hash`; if it
    /// can, its first lines are asked for.
    #[inline(always)]
    pub(super) fn may_be(self, hash: u64) -> bool {
        if self.0 >> 52 != Self::fragment(hash) {
            return false;
        }
        let lines = HINT_LINES[((self.0 >> ADDRESS_BITS) & 0xf) as usize] as usize;
        prefetch((self.0 & ADDRESS) as *const u8, lines.min(PREFETCH_LINES));
        true
    }
}

/// Ask for `lines` cache lines from `at`. A hint: nothing is read.
#[inline(always)]
fn prefetch(at: *const u8, lines: usize) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    for line in 0..lines {
        // SAFETY: a prefetch reads nothing and faults on nothing.
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                at.wrapping_add(line * 64) as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            )
        };
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    for line in 0..lines {
        // SAFETY: as above.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) at.wrapping_add(line * 64),
                options(nostack, readonly, preserves_flags)
            )
        };
    }
    #[cfg(any(miri, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
    let _ = (at, lines);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "above 2^48")]
    fn a_slot_refuses_an_address_past_48_bits() {
        // Refused before the entry is looked at: the address is never read.
        let far = Entry(NonNull::new(std::ptr::without_provenance_mut::<Header>(1 << 50)).unwrap());
        let _ = Slot::new(far, 0);
    }

    #[test]
    fn a_slot_keeps_its_entry_fragment_and_hint() {
        for (klen, vlen) in [(1, 0), (15, 100), (15, 1000), (300, 3000), (15, 70_000)] {
            let key = vec![b'k'; klen];
            let entry = Entry::string(&key, &vec![7u8; vlen]);
            let hash = 0xabc_u64 << 32 | 0x1234_5678;
            let slot = Slot::new(entry, hash);
            assert!(slot.entry() == entry);
            assert!(slot.may_be(hash));
            assert!(!slot.may_be(hash ^ 1 << 40));
            // The low bits are not the fragment's: hashbrown's index.
            assert!(slot.may_be(hash ^ 0xffff));
            let lines = HINT_LINES[((slot.0 >> ADDRESS_BITS) & 0xf) as usize] as usize;
            let need = entry.size().div_ceil(64);
            assert!(lines >= need.min(64), "{lines} lines for {need}");
            let moved = Entry::string(&key, &vec![7u8; vlen]);
            let again = slot.moved_to(moved);
            assert!(again.entry() == moved);
            assert_eq!(again.0 >> ADDRESS_BITS, slot.0 >> ADDRESS_BITS);
            // SAFETY: each made here with one hold, used no more.
            unsafe {
                entry.release();
                moved.release();
            }
        }
    }

    #[test]
    fn the_header_is_twelve_bytes_and_the_key_follows() {
        assert_eq!(HEADER, 12);
        let entry = Entry::string(b"key", b"value");
        assert_eq!(entry.size(), 12 + 3 + 5);
        assert_eq!(entry.key(), b"key");
        assert_eq!(entry.form(), INLINE);
        assert_eq!(entry.holders(), 1);
        // SAFETY: made here with one hold, used no more.
        unsafe { entry.release() };
        let long = vec![b'x'; 255];
        let entry = Entry::new(&long, KiviDBValue::List(Default::default()));
        assert_eq!(entry.key(), &long[..]);
        assert_eq!(entry.form(), WHOLE);
        assert_eq!(entry.size() % 8, 0);
        // SAFETY: as above.
        unsafe { entry.release() };
    }
}
