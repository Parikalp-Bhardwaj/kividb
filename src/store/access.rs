//! A key's use, written by the lookup its command does anyway.
//!
//! An eviction policy that goes by how keys are used (LRU, LFU) has each
//! key's access word brought up to date whenever a command uses the key
//! (see `eviction::Access`). Done after the command, that was a second
//! lookup of each of its keys: the shard chosen again, its lock taken again,
//! the key hashed and found again, about a tenth of what a GET costs. Here
//! it is part of the command's own lookup.
//!
//! The command's dispatcher opens an [`AccessScope`] around the command,
//! saying what a use writes (a [`stamp`]); while it is open, the lookups of
//! this thread's commands ([`super::ShardMap::get`], `get_expiring`,
//! `contains_key`, every write of a key's value, `set_expires`, `persist`)
//! write it to the word of the entry they find or make. Lookups that only
//! ask about a key (`expires`, `access`, `peek`, the iterations, samples and
//! expiry passes) never do, scope or not. A command that does not count as
//! a use (`OBJECT`, `TYPE`, `EXISTS`, ...) opens a scope that writes
//! nothing, so that nothing it calls writes either.
//!
//! **Stamps.** 0 writes nothing. A stamp with its top bit clear is the word
//! itself, written as it is when the entry's differs: when the key was last
//! used (LRU). One with the top bit set (LFU) is handed, with the entry's
//! word, to `eviction::counted`, which gives the word after one more use;
//! and since that is a count, each key is counted once a command however
//! many times the command looks it up (INCR reads, then writes). A few keys
//! a command are remembered for that; past them a key may be counted twice.
//!
//! The scope is thread-local: a lookup pays one load of it, and, with no
//! policy that goes by use, nothing else.

use std::cell::Cell;

use super::entry::Entry;

/// A stamp's top bit: what follows is an LFU stamp, not a word.
pub const COUNTED: u32 = 1 << 31;

/// Keys remembered as counted in one scope.
const SEEN: usize = 6;

#[derive(Clone, Copy)]
struct Seen {
    len: u8,
    keys: [u64; SEEN],
}

impl Seen {
    const NONE: Seen = Seen {
        len: 0,
        keys: [0; SEEN],
    };
}

thread_local! {
    static STAMP: Cell<u32> = const { Cell::new(0) };
    static COUNTED_KEYS: Cell<Seen> = const { Cell::new(Seen::NONE) };
}

/// What a lookup on this thread writes to the access word of the entry it
/// finds: 0 for nothing.
#[inline(always)]
pub(super) fn stamp() -> u32 {
    STAMP.with(Cell::get)
}

/// While this lives, lookups on this thread write `stamp` (see the module):
/// what a command's dispatcher opens around it.
pub struct AccessScope {
    before: u32,
    seen: Option<Seen>,
}

impl AccessScope {
    /// Lookups write `stamp`. A scope inside another (a command that runs
    /// others) puts the outer one back when it goes.
    #[inline]
    pub fn open(stamp: u32) -> Self {
        let before = STAMP.with(|s| s.replace(stamp));
        let seen = (stamp & COUNTED != 0).then(|| COUNTED_KEYS.with(|s| s.replace(Seen::NONE)));
        AccessScope { before, seen }
    }

    /// Lookups write nothing: for a command that only asks about keys.
    #[inline]
    pub fn none() -> Self {
        Self::open(0)
    }

    /// Whether lookups on this thread write a key's use now: for a command
    /// that can answer without looking its key up.
    #[inline]
    pub fn active() -> bool {
        stamp() != 0
    }
}

impl Drop for AccessScope {
    #[inline]
    fn drop(&mut self) {
        STAMP.with(|s| s.set(self.before));
        if let Some(seen) = self.seen {
            COUNTED_KEYS.with(|s| s.set(seen));
        }
    }
}

/// `entry` was found or made by a lookup under the stamp `stamp` (not 0);
/// `key` tells its key from others (the table and the hash). The shard is
/// locked, for reading at least.
#[inline(always)]
pub(super) fn used(entry: Entry, stamp: u32, key: u64) {
    if stamp & COUNTED == 0 {
        // Written only when it changes: a key read again within the same
        // second leaves its line alone.
        if entry.access() != stamp {
            entry.set_access(stamp);
        }
    } else {
        counted(entry, stamp, key);
    }
}

#[inline(never)]
fn counted(entry: Entry, stamp: u32, key: u64) {
    let first = COUNTED_KEYS.with(|cell| {
        let mut seen = cell.get();
        let n = seen.len as usize;
        if seen.keys[..n].contains(&key) {
            return false;
        }
        if n < SEEN {
            seen.keys[n] = key;
            seen.len += 1;
            cell.set(seen);
        }
        true
    });
    if first {
        let word = entry.access();
        let next = crate::eviction::counted(word, stamp);
        if next != word {
            entry.set_access(next);
        }
    }
}
