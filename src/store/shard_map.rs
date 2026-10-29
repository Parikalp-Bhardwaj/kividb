//! One shard of a database's keyspace, behind an interface that does not
//! say how it is kept.
//!
//! Commands ask for a key's value and get a [`ValueRef`], a view of it that
//! lives as long as the shard's lock. They never see a `&KiviDBValue` from
//! the table. A value going in, coming out or moving between keys is an
//! owned [`KiviDBValue`] (or, coming out, a [`Taken`] entry that can give
//! one).
//!
//! The table is hashbrown's, used directly: its buckets can be reached by
//! index, which is what a cursor that survives the lock being let go
//! (compaction), and a key picked at random (`RANDOMKEY`, eviction), need.
//! Each bucket is an eight-byte [`Slot`]: the address of one block holding
//! the key and its value together (`entry`).

use std::collections::VecDeque;
use std::sync::Arc;

use hashbrown::HashTable;
use parking_lot::RwLock;

use super::entry::{Entry, Left, Relocation, Slot};
pub use super::entry::{EntryRef, Taken};
use crate::shared_bytes::SharedBytes;
use crate::types::{
    ArrayInner, HashFields, HashInner, KiviDBValue, SortedSetInner, StreamInner, TsInner,
    VectorSetInner,
};

/// A key's string value, borrowed from its shard: its bytes, kept in the
/// key's entry, or a `SharedBytes` of their own when they are too long.
#[derive(Clone, Copy)]
pub struct StrRef<'a>(Bytes<'a>);

#[derive(Clone, Copy)]
enum Bytes<'a> {
    Inline(&'a [u8]),
    Shared(&'a SharedBytes),
}

impl<'a> StrRef<'a> {
    #[inline]
    pub(super) fn bytes(bytes: &'a [u8]) -> Self {
        StrRef(Bytes::Inline(bytes))
    }

    #[inline]
    pub fn as_bytes(self) -> &'a [u8] {
        match self.0 {
            Bytes::Inline(bytes) => bytes,
            Bytes::Shared(shared) => shared.as_slice(),
        }
    }

    /// The bytes, to keep once the shard is let go of: a second hold on
    /// the same block when they have one of their own, a copy when they are
    /// in the key's entry.
    #[inline]
    pub fn to_shared(self) -> SharedBytes {
        match self.0 {
            Bytes::Inline(bytes) => SharedBytes::from(bytes),
            Bytes::Shared(shared) => shared.clone(),
        }
    }

    /// The `SharedBytes` the bytes are in, when they are not in the entry.
    #[inline]
    pub(super) fn shared(self) -> Option<&'a SharedBytes> {
        match self.0 {
            Bytes::Inline(_) => None,
            Bytes::Shared(shared) => Some(shared),
        }
    }

    /// Whether `other` is this very value, not merely equal bytes. Bytes
    /// in an entry are no `SharedBytes`: never.
    #[inline]
    pub fn is(self, other: &SharedBytes) -> bool {
        self.shared()
            .is_some_and(|shared| SharedBytes::ptr_eq(shared, other))
    }
}

impl std::ops::Deref for StrRef<'_> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl AsRef<[u8]> for StrRef<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// A key's value, borrowed from its shard: what [`ShardMap::get`] gives.
/// The variants are [`KiviDBValue`]'s. A collection is its shared handle,
/// which a command locks, or clones to keep.
#[derive(Clone, Copy)]
pub enum ValueRef<'a> {
    String(StrRef<'a>),
    List(&'a Arc<RwLock<VecDeque<Arc<[u8]>>>>),
    Hash(&'a Arc<RwLock<HashFields>>),
    Json(&'a Arc<RwLock<serde_json::Value>>),
    Set(&'a Arc<RwLock<crate::member_set::MemberSet>>),
    VectorSet(&'a Arc<RwLock<VectorSetInner>>),
    SortedSet(&'a Arc<RwLock<SortedSetInner>>),
    Stream(&'a Arc<RwLock<StreamInner>>),
    TimeSeries(&'a Arc<RwLock<TsInner>>),
    HashEx(&'a Arc<RwLock<HashInner>>),
    Array(&'a Arc<RwLock<ArrayInner>>),
}

impl<'a> From<&'a KiviDBValue> for ValueRef<'a> {
    #[inline]
    fn from(value: &'a KiviDBValue) -> Self {
        match value {
            KiviDBValue::String(v) => ValueRef::String(StrRef(Bytes::Shared(v))),
            KiviDBValue::List(v) => ValueRef::List(v),
            KiviDBValue::Hash(v) => ValueRef::Hash(v),
            KiviDBValue::Json(v) => ValueRef::Json(v),
            KiviDBValue::Set(v) => ValueRef::Set(v),
            KiviDBValue::VectorSet(v) => ValueRef::VectorSet(v),
            KiviDBValue::SortedSet(v) => ValueRef::SortedSet(v),
            KiviDBValue::Stream(v) => ValueRef::Stream(v),
            KiviDBValue::TimeSeries(v) => ValueRef::TimeSeries(v),
            KiviDBValue::HashEx(v) => ValueRef::HashEx(v),
            KiviDBValue::Array(v) => ValueRef::Array(v),
        }
    }
}

impl KiviDBValue {
    /// A view of an owned value, as [`ShardMap::get`] gives one of a stored
    /// value: for code that takes either.
    #[inline]
    pub fn view(&self) -> ValueRef<'_> {
        ValueRef::from(self)
    }
}

impl ValueRef<'_> {
    /// The value, to keep once the shard is let go of: another handle on
    /// the same string or collection (see [`KiviDBValue::deep_clone`] for a
    /// copy that shares nothing).
    pub fn to_value(self) -> KiviDBValue {
        match self {
            ValueRef::String(v) => KiviDBValue::String(v.to_shared()),
            ValueRef::List(v) => KiviDBValue::List(Arc::clone(v)),
            ValueRef::Hash(v) => KiviDBValue::Hash(Arc::clone(v)),
            ValueRef::Json(v) => KiviDBValue::Json(Arc::clone(v)),
            ValueRef::Set(v) => KiviDBValue::Set(Arc::clone(v)),
            ValueRef::VectorSet(v) => KiviDBValue::VectorSet(Arc::clone(v)),
            ValueRef::SortedSet(v) => KiviDBValue::SortedSet(Arc::clone(v)),
            ValueRef::Stream(v) => KiviDBValue::Stream(Arc::clone(v)),
            ValueRef::TimeSeries(v) => KiviDBValue::TimeSeries(Arc::clone(v)),
            ValueRef::HashEx(v) => KiviDBValue::HashEx(Arc::clone(v)),
            ValueRef::Array(v) => KiviDBValue::Array(Arc::clone(v)),
        }
    }

    /// [`KiviDBValue::deep_clone`] of the value.
    pub fn deep_clone(self) -> KiviDBValue {
        self.to_value().deep_clone()
    }

    /// [`KiviDBValue::is_empty_collection`].
    pub fn is_empty_collection(self) -> bool {
        match self {
            ValueRef::List(list) => list.read().is_empty(),
            ValueRef::Set(set) => set.read().is_empty(),
            ValueRef::SortedSet(zset) => zset.read().is_empty(),
            ValueRef::Hash(hash) => hash.read().is_empty(),
            ValueRef::HashEx(hash) => hash.read().is_empty(),
            ValueRef::VectorSet(vset) => vset.read().entries.is_empty(),
            ValueRef::String(_)
            | ValueRef::Json(_)
            | ValueRef::Stream(_)
            | ValueRef::TimeSeries(_)
            | ValueRef::Array(_) => false,
        }
    }

    /// [`KiviDBValue::type_name`].
    pub fn type_name(self) -> &'static str {
        match self {
            ValueRef::String(_) => "string",
            ValueRef::List(_) => "list",
            ValueRef::Hash(_) => "hash",
            ValueRef::Json(_) => "ReJSON-RL",
            ValueRef::Set(_) => "set",
            ValueRef::VectorSet(_) => "vectorset",
            ValueRef::SortedSet(_) => "zset",
            ValueRef::Stream(_) => "stream",
            ValueRef::TimeSeries(_) => "TSDB-TYPE",
            ValueRef::HashEx(_) => "hash",
            ValueRef::Array(_) => "array",
        }
    }
}

/// What [`ShardMap::relocate_at`] did.
pub enum Relocated {
    /// Moved: what was left behind, to be let go of once the shard is
    /// unlocked (freeing it can give a slab back to the kernel).
    Moved(Leftover),
    /// In a slab being emptied, but held by something besides its key.
    Held,
    /// Nothing there, or nothing in a slab being emptied.
    Stays,
}

/// A block compaction moved a key's entry, or its string, out of.
pub enum Leftover {
    Entry(Left),
    Bytes(SharedBytes),
}

/// One shard's keys and values: a table of [`Slot`]s, each the address of
/// an entry holding a key and its value (`entry`).
pub struct ShardMap {
    table: HashTable<Slot>,
    hash: ahash::RandomState,
}

impl Default for ShardMap {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ShardMap {
    fn drop(&mut self) {
        self.clear();
    }
}

impl ShardMap {
    pub fn new() -> Self {
        ShardMap {
            table: HashTable::new(),
            hash: ahash::RandomState::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        ShardMap {
            table: HashTable::with_capacity(capacity),
            hash: ahash::RandomState::new(),
        }
    }

    #[inline(always)]
    fn hash_of(&self, key: &[u8]) -> u64 {
        self.hash.hash_one(key)
    }

    /// The slot of `key`, hashed `hash`.
    #[inline(always)]
    fn find(&self, hash: u64, key: &[u8]) -> Option<Slot> {
        self.table
            .find(hash, |s| s.may_be(hash) && s.entry().has_key(key))
            .copied()
    }

    #[inline(always)]
    pub fn get(&self, key: &str) -> Option<ValueRef<'_>> {
        let key = key.as_bytes();
        self.find(self.hash_of(key), key).map(|s| s.entry().value())
    }

    #[inline(always)]
    pub fn contains_key(&self, key: &str) -> bool {
        let key = key.as_bytes();
        self.find(self.hash_of(key), key).is_some()
    }

    /// A hold on `key`'s entry: its key and value, readable once the shard
    /// is let go of, and unchanged for as long as it is held.
    pub fn get_ref(&self, key: &str) -> Option<EntryRef> {
        let key = key.as_bytes();
        // SAFETY: the entry is in this table, which the caller has locked.
        self.find(self.hash_of(key), key)
            .map(|s| unsafe { EntryRef::new(s.entry()) })
    }

    /// Whether `key`'s entry is the one `held` holds: whether the key's
    /// value is the very one it had when `held` was taken.
    pub fn holds(&self, key: &str, held: &EntryRef) -> bool {
        let key = key.as_bytes();
        self.find(self.hash_of(key), key)
            .is_some_and(|s| held.is(s.entry()))
    }

    /// Put `entry` in the table under its key, and give back what the key
    /// had. One probe, whether the key is there or not.
    fn put(&mut self, entry: Entry) -> Option<Taken> {
        let key = entry.key();
        let hash = self.hash_of(key);
        let slot = Slot::new(entry, hash);
        let hasher = &self.hash;
        match self.table.entry(
            hash,
            |s| s.may_be(hash) && s.entry().has_key(key),
            |s| hasher.hash_one(s.entry().key()),
        ) {
            hashbrown::hash_table::Entry::Occupied(mut at) => {
                let old = std::mem::replace(at.get_mut(), slot).entry();
                // SAFETY: the table's hold on the old entry goes to the caller.
                Some(unsafe { Taken::new(old) })
            }
            hashbrown::hash_table::Entry::Vacant(place) => {
                place.insert(slot);
                None
            }
        }
    }

    /// Set `key` to `value`, and give back what it had: dropping that frees
    /// it, which is best done once the shard is unlocked.
    #[inline]
    pub fn insert(&mut self, key: impl AsRef<str>, value: KiviDBValue) -> Option<Taken> {
        self.put(Entry::new(key.as_ref().as_bytes(), value))
    }

    /// Set `key` to the string `bytes`: [`ShardMap::insert`] of a string,
    /// with its bytes copied once, into the entry.
    #[inline]
    pub fn insert_str(&mut self, key: impl AsRef<str>, bytes: impl AsRef<[u8]>) -> Option<Taken> {
        self.put(Entry::string(key.as_ref().as_bytes(), bytes.as_ref()))
    }

    /// Put an entry made before the shard was locked ([`NewEntry`]).
    #[inline]
    pub fn insert_entry(&mut self, entry: NewEntry) -> Option<Taken> {
        let at = entry.0;
        std::mem::forget(entry);
        self.put(at)
    }

    #[inline]
    pub fn remove(&mut self, key: &str) -> Option<Taken> {
        let key = key.as_bytes();
        let hash = self.hash_of(key);
        match self
            .table
            .find_entry(hash, |s| s.may_be(hash) && s.entry().has_key(key))
        {
            // SAFETY: the table's hold goes to the caller.
            Ok(at) => Some(unsafe { Taken::new(at.remove().0.entry()) }),
            Err(_) => None,
        }
    }

    /// The value of `key`, made with `make` if there is none.
    pub fn get_or_insert_with(
        &mut self,
        key: &str,
        make: impl FnOnce() -> KiviDBValue,
    ) -> ValueRef<'_> {
        if !self.contains_key(key) {
            self.insert(key, make());
        }
        self.get(key).expect("a key just put in")
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, ValueRef<'_>)> {
        self.table.iter().map(|s| {
            let e = s.entry();
            (e.key_str(), e.value())
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.table.iter().map(|s| s.entry().key_str())
    }

    pub fn values(&self) -> impl Iterator<Item = ValueRef<'_>> {
        self.table.iter().map(|s| s.entry().value())
    }

    /// Every entry, each as a key, a value and the means to hold it.
    pub fn entries(&self) -> impl Iterator<Item = EntryView<'_>> {
        self.table.iter().map(|s| EntryView {
            entry: s.entry(),
            _map: std::marker::PhantomData,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Keys the table has room for before it grows.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Take every key out, keeping the table's room.
    pub fn clear(&mut self) {
        for slot in self.table.drain() {
            // SAFETY: the table's hold, given up.
            unsafe { slot.entry().release() };
        }
    }

    pub fn shrink_to(&mut self, keys: usize) {
        let hasher = &self.hash;
        self.table
            .shrink_to(keys, |s| hasher.hash_one(s.entry().key()));
    }

    /// How many buckets the table has: the positions [`ShardMap::bucket`]
    /// takes. They stay where they are while keys come and go, until the
    /// table grows or shrinks.
    #[inline]
    pub fn buckets(&self) -> usize {
        self.table.num_buckets()
    }

    /// The key and value in bucket `index`, if it holds one.
    #[inline]
    pub fn bucket(&self, index: usize) -> Option<(&str, ValueRef<'_>)> {
        self.table.get_bucket(index).map(|s| {
            let e = s.entry();
            (e.key_str(), e.value())
        })
    }

    /// Up to `count` keys and values from the buckets at and after `start`
    /// (taken modulo the bucket count), wrapping round once: what a sample
    /// from a random place in the table is. Costs about `count` divided by
    /// how full the table is, where walking to the n-th key cost n.
    pub fn sample_from(
        &self,
        start: usize,
        count: usize,
    ) -> impl Iterator<Item = (&str, ValueRef<'_>)> {
        let buckets = self.buckets();
        let start = if buckets == 0 { 0 } else { start % buckets };
        (0..buckets)
            .filter_map(move |step| self.bucket((start + step) % buckets))
            .take(if self.is_empty() { 0 } else { count })
    }

    /// One key picked at random, `pick()` giving the random numbers. A
    /// random bucket is tried until one is full, as Redis's
    /// `dictGetFairRandomKey` does; a table that is very sparse is walked
    /// from a random bucket.
    pub fn random_entry(&self, mut pick: impl FnMut() -> usize) -> Option<(&str, ValueRef<'_>)> {
        if self.is_empty() {
            return None;
        }
        let buckets = self.buckets();
        for _ in 0..64 {
            if let Some(found) = self.bucket(pick() % buckets) {
                return Some(found);
            }
        }
        self.sample_from(pick(), 1).next()
    }

    /// Whether the entry in bucket `index`, or its string's bytes, is in a
    /// slab compaction is emptying (`regions` is
    /// `value_slab::regions_snapshot`).
    pub fn evacuating_at(&self, index: usize, regions: &[usize]) -> bool {
        let Some(slot) = self.table.get_bucket(index) else {
            return false;
        };
        let entry = slot.entry();
        (crate::value_slab::class_of(entry.size()).is_some()
            && crate::value_slab::is_evacuating(entry.base(), regions))
            || matches!(entry.value(), ValueRef::String(s)
                if s.shared().is_some_and(|b| crate::value_slab::is_evacuating(b.block(), regions)))
    }

    /// Move the entry in bucket `index`, or its string's bytes, out of a
    /// slab compaction is emptying, if the table's is its only hold.
    pub fn relocate_at(&mut self, index: usize, regions: &[usize]) -> Relocated {
        let Some(slot) = self.table.get_bucket_mut(index) else {
            return Relocated::Stays;
        };
        let entry = slot.entry();
        // SAFETY: the table is locked for writing (`&mut self`), and holds
        // the entry.
        match unsafe { entry.relocate(regions) } {
            Relocation::Stays => Relocated::Stays,
            Relocation::Held => Relocated::Held,
            Relocation::Moved { to, left } => {
                // The hash is not stored; the fragment is in the slot.
                *slot = slot.moved_to(to);
                Relocated::Moved(Leftover::Entry(left))
            }
            Relocation::BytesMoved(old) => Relocated::Moved(Leftover::Bytes(old)),
        }
    }
}

/// A key's entry, made before its shard is locked: what the shard then
/// waits for is only the table. Freed if dropped unused.
pub struct NewEntry(Entry);

impl NewEntry {
    pub fn new(key: &str, value: KiviDBValue) -> Self {
        NewEntry(Entry::new(key.as_bytes(), value))
    }

    /// The string `bytes`, copied once, into the entry.
    pub fn string(key: &str, bytes: &[u8]) -> Self {
        NewEntry(Entry::string(key.as_bytes(), bytes))
    }
}

impl Drop for NewEntry {
    fn drop(&mut self) {
        // SAFETY: the only hold, never put in a table.
        unsafe { self.0.release() };
    }
}

/// One entry of a locked shard, as [`ShardMap::entries`] gives them.
pub struct EntryView<'a> {
    entry: Entry,
    _map: std::marker::PhantomData<&'a ShardMap>,
}

impl<'a> EntryView<'a> {
    pub fn key(&self) -> &'a str {
        self.entry.key_str()
    }

    pub fn value(&self) -> ValueRef<'a> {
        self.entry.value()
    }

    /// A hold on the entry, to read it once the shard is let go of.
    pub fn hold(&self) -> EntryRef {
        // SAFETY: the shard is locked for as long as `'a`.
        unsafe { EntryRef::new(self.entry) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> KiviDBValue {
        KiviDBValue::String(SharedBytes::from(v))
    }

    fn text(v: Option<ValueRef<'_>>) -> Option<Vec<u8>> {
        match v {
            Some(ValueRef::String(b)) => Some(b.to_vec()),
            _ => None,
        }
    }

    #[test]
    fn insert_get_remove() {
        let mut map = ShardMap::new();
        assert!(map.insert("a", s("1")).is_none());
        assert_eq!(text(map.get("a")), Some(b"1".to_vec()));
        assert!(map.contains_key("a"));
        let old = map.insert(crate::store::Key::from("a"), s("2"));
        assert!(
            matches!(old.map(Taken::into_value), Some(KiviDBValue::String(b)) if &b[..] == b"1")
        );
        assert_eq!(map.len(), 1);
        assert!(map.remove("a").is_some());
        assert!(map.remove("a").is_none());
        assert!(map.is_empty());
    }

    #[test]
    fn keys_survive_growth_and_shrinking() {
        let mut map = ShardMap::new();
        for i in 0..10_000 {
            map.insert(format!("key-{i}"), s(&i.to_string()));
        }
        for i in (0..10_000).filter(|i| i % 10 != 0) {
            map.remove(&format!("key-{i}"));
        }
        map.shrink_to(1_000);
        assert_eq!(map.len(), 1_000);
        assert_eq!(text(map.get("key-990")), Some(b"990".to_vec()));
        assert!(map.get("key-991").is_none());
    }

    #[test]
    fn buckets_reach_every_key_once() {
        let mut map = ShardMap::new();
        for i in 0..1_000 {
            map.insert(format!("k{i}"), s("v"));
        }
        let mut seen: Vec<String> = (0..map.buckets())
            .filter_map(|i| map.bucket(i).map(|(k, _)| k.to_string()))
            .collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 1_000);
        assert!(map.bucket(map.buckets()).is_none());
    }

    #[test]
    fn samples_and_random_keys() {
        let mut map = ShardMap::new();
        assert!(map.random_entry(|| 7).is_none());
        assert_eq!(map.sample_from(3, 5).count(), 0);
        for i in 0..100 {
            map.insert(format!("k{i}"), s("v"));
        }
        assert_eq!(map.sample_from(12345, 5).count(), 5);
        assert_eq!(map.sample_from(12345, 500).count(), 100);
        let mut n = 0usize;
        let (key, _) = map
            .random_entry(|| {
                n += 7919;
                n
            })
            .unwrap();
        assert!(map.contains_key(key));
        // A table nearly empty after a big delete still gives its key.
        for i in 1..100 {
            map.remove(&format!("k{i}"));
        }
        assert_eq!(map.random_entry(|| 0).map(|(k, _)| k), Some("k0"));
    }

    #[test]
    fn get_or_insert_with_makes_once() {
        let mut map = ShardMap::new();
        let mut made = 0;
        for _ in 0..3 {
            map.get_or_insert_with("a", || {
                made += 1;
                s("x")
            });
        }
        assert_eq!(made, 1);
        assert_eq!(map.len(), 1);
    }
}
