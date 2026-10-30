// TAG and NUMERIC postings: which documents hold which tag, and which hold a
// number in a range, without reading the documents.
//
// Each document an index covers is given a small number, its id, for as long
// as it is in the index: ids are what the sets below hold, so that a set of
// documents is a bitmap or a sorted list of `u32`s, and two of them are
// intersected word by word. A TAG field keeps, for each tag, the set of
// documents that hold it; a NUMERIC field keeps its values sorted, so that a
// range is read off in order. Both keep what each document put in, so that a
// document written again, or gone, is taken out of exactly the sets it was in.
//
// A query's TAG and NUMERIC clauses, and `AND`, `OR` and `NOT` of them, are
// then a set of ids computed from these (see `commands::search::plan`),
// instead of a filter tried on every document. An index without a TAG or a
// NUMERIC field keeps none of this, and gives out no ids.
//
// What a document holds is read as the filter reads it (`filter::eval`): a
// TAG value split at the field's separator, trimmed, lower-cased unless the
// field is CASESENSITIVE; a NUMERIC value as the number its text parses to.
// The two must agree, or a query would answer differently with postings than
// without them.

use super::{FieldType, SchemaField};
use ahash::AHashMap;
use std::collections::BTreeSet;
use std::sync::Arc;

/// A node of a graph whose document id is not known (yet): see
/// [`super::HnswNode::doc`].
pub const NO_DOC: u32 = u32::MAX;

// ── Bitmaps ──────────────────────────────────────────────────────────────────

/// A set of document ids, one bit each.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Bits {
    words: Vec<u64>,
}

impl Bits {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn set(&mut self, id: u32) {
        let (w, b) = ((id / 64) as usize, id % 64);
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        self.words[w] |= 1 << b;
    }

    #[inline]
    pub fn unset(&mut self, id: u32) {
        let (w, b) = ((id / 64) as usize, id % 64);
        if let Some(word) = self.words.get_mut(w) {
            *word &= !(1 << b);
        }
    }

    #[inline]
    pub fn get(&self, id: u32) -> bool {
        let (w, b) = ((id / 64) as usize, id % 64);
        self.words.get(w).is_some_and(|word| word & (1 << b) != 0)
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    pub fn and_with(&mut self, other: &Bits) {
        self.words.truncate(other.words.len());
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= b;
        }
    }

    pub fn or_with(&mut self, other: &Bits) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// Take out of this set every id that is in `other`.
    pub fn and_not(&mut self, other: &Bits) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= !b;
        }
    }

    /// The ids, in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                if rest == 0 {
                    return None;
                }
                let b = rest.trailing_zeros();
                rest &= rest - 1;
                Some(w as u32 * 64 + b)
            })
        })
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.words.capacity() * 8
    }
}

/// The documents that hold one tag: a sorted list while they are few, a
/// bitmap once the list would be the larger of the two.
#[derive(Debug, Clone)]
enum IdSet {
    Sparse(Vec<u32>),
    Dense { bits: Bits, len: usize },
}

impl Default for IdSet {
    fn default() -> Self {
        IdSet::Sparse(Vec::new())
    }
}

impl IdSet {
    fn len(&self) -> usize {
        match self {
            IdSet::Sparse(ids) => ids.len(),
            IdSet::Dense { len, .. } => *len,
        }
    }

    /// Add `id`. `slots` is how many ids there are, which a bitmap costs.
    fn insert(&mut self, id: u32, slots: usize) {
        match self {
            IdSet::Sparse(ids) => {
                if let Err(at) = ids.binary_search(&id) {
                    ids.insert(at, id);
                }
                // A list of n ids costs 32n bits; a bitmap one bit a slot.
                if ids.len() >= 64 && ids.len() * 32 > slots {
                    let mut bits = Bits::new();
                    for &id in ids.iter() {
                        bits.set(id);
                    }
                    *self = IdSet::Dense {
                        len: ids.len(),
                        bits,
                    };
                }
            }
            IdSet::Dense { bits, len } => {
                if !bits.get(id) {
                    bits.set(id);
                    *len += 1;
                }
            }
        }
    }

    fn remove(&mut self, id: u32, slots: usize) {
        match self {
            IdSet::Sparse(ids) => {
                if let Ok(at) = ids.binary_search(&id) {
                    ids.remove(at);
                }
            }
            IdSet::Dense { bits, len } => {
                if bits.get(id) {
                    bits.unset(id);
                    *len -= 1;
                }
                // Back to a list well below the point it became a bitmap, so
                // that a set at the boundary does not flip back and forth.
                if *len * 64 < slots {
                    *self = IdSet::Sparse(bits.iter().collect());
                }
            }
        }
    }

    fn or_into(&self, out: &mut Bits) {
        match self {
            IdSet::Sparse(ids) => ids.iter().for_each(|&id| out.set(id)),
            IdSet::Dense { bits, .. } => out.or_with(bits),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            IdSet::Sparse(ids) => ids.capacity() * 4,
            IdSet::Dense { bits, .. } => bits.bytes(),
        }
    }
}

// ── TAG ──────────────────────────────────────────────────────────────────────

/// What one document put into a TAG field: the ids of its tags.
#[derive(Debug, Clone, Default)]
enum Held {
    #[default]
    Nothing,
    One(u32),
    Many(Box<[u32]>),
}

impl Held {
    fn from_vec(mut tags: Vec<u32>) -> Self {
        tags.sort_unstable();
        tags.dedup();
        match tags.as_slice() {
            [] => Held::Nothing,
            [one] => Held::One(*one),
            _ => Held::Many(tags.into_boxed_slice()),
        }
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            Held::Nothing => &[],
            Held::One(one) => std::slice::from_ref(one),
            Held::Many(many) => many,
        }
    }
}

#[derive(Debug, Clone)]
struct TagPostings {
    separator: char,
    case_sensitive: bool,
    /// Tag → its number, by which the sets and documents refer to it.
    numbers: AHashMap<Box<str>, u32>,
    /// By tag number: the tag and the documents holding it. A tag no
    /// document holds any more gives its number back (`free`).
    tags: Vec<(Box<str>, IdSet)>,
    free: Vec<u32>,
    /// By document id: what it holds.
    docs: Vec<Held>,
}

impl TagPostings {
    fn new(separator: char, case_sensitive: bool) -> Self {
        Self {
            separator,
            case_sensitive,
            numbers: AHashMap::new(),
            tags: Vec::new(),
            free: Vec::new(),
            docs: Vec::new(),
        }
    }

    /// The tags of a value, as `filter::eval` reads them.
    fn parse(&self, value: Option<&[u8]>) -> Vec<String> {
        let Some(text) = value.and_then(|v| std::str::from_utf8(v).ok()) else {
            return Vec::new();
        };
        super::filter::split_tags(text, self.separator)
            .map(|tag| {
                if self.case_sensitive {
                    tag.to_string()
                } else {
                    tag.to_lowercase()
                }
            })
            .collect()
    }

    fn number_of(&mut self, tag: String) -> u32 {
        if let Some(&n) = self.numbers.get(tag.as_str()) {
            return n;
        }
        let tag: Box<str> = tag.into_boxed_str();
        let n = match self.free.pop() {
            Some(n) => {
                self.tags[n as usize] = (tag.clone(), IdSet::default());
                n
            }
            None => {
                self.tags.push((tag.clone(), IdSet::default()));
                (self.tags.len() - 1) as u32
            }
        };
        self.numbers.insert(tag, n);
        n
    }

    fn set(&mut self, doc: u32, value: Option<&[u8]>, slots: usize) {
        let tags = self.parse(value);
        let numbers: Vec<u32> = tags.into_iter().map(|t| self.number_of(t)).collect();
        let new = Held::from_vec(numbers);
        let at = doc as usize;
        if at >= self.docs.len() {
            self.docs.resize(at + 1, Held::Nothing);
        }
        let old = std::mem::replace(&mut self.docs[at], new);
        let new = self.docs[at].as_slice().to_vec();
        for &n in old.as_slice() {
            if !new.contains(&n) {
                self.drop_from(n, doc, slots);
            }
        }
        for &n in &new {
            if !old.as_slice().contains(&n) {
                self.tags[n as usize].1.insert(doc, slots);
            }
        }
    }

    fn drop_from(&mut self, n: u32, doc: u32, slots: usize) {
        let set = &mut self.tags[n as usize].1;
        set.remove(doc, slots);
        if set.len() == 0 {
            self.release(n);
        }
    }

    fn release(&mut self, n: u32) {
        let (tag, _) = std::mem::take(&mut self.tags[n as usize]);
        if self.numbers.remove(&tag).is_some() {
            self.free.push(n);
        }
    }

    fn clear(&mut self, doc: u32, slots: usize) {
        if let Some(held) = self.docs.get_mut(doc as usize) {
            let old = std::mem::take(held);
            for &n in old.as_slice() {
                self.drop_from(n, doc, slots);
            }
        }
    }

    /// The documents holding any of `values`, or a tag starting with any of
    /// `prefixes`.
    fn any_of(&self, values: &[String], prefixes: &[String], out: &mut Bits) {
        for value in values {
            if let Some(&n) = self.numbers.get(value.as_str()) {
                self.tags[n as usize].1.or_into(out);
            }
        }
        if !prefixes.is_empty() {
            for (tag, &n) in &self.numbers {
                if prefixes.iter().any(|p| tag.starts_with(p.as_str())) {
                    self.tags[n as usize].1.or_into(out);
                }
            }
        }
    }

    fn bytes(&self) -> usize {
        let tags: usize = self
            .tags
            .iter()
            .map(|(tag, set)| 24 + tag.len() + set.bytes())
            .sum();
        let numbers =
            self.numbers.capacity() * 24 + self.numbers.keys().map(|k| k.len()).sum::<usize>();
        let docs = self.docs.capacity() * std::mem::size_of::<Held>()
            + self
                .docs
                .iter()
                .map(|h| match h {
                    Held::Many(m) => m.len() * 4,
                    _ => 0,
                })
                .sum::<usize>();
        tags + numbers + docs
    }
}

// ── NUMERIC ──────────────────────────────────────────────────────────────────

/// `v` as a key that sorts as the number does. Minus zero is zero: the two
/// are equal to a comparison, and so to a range.
#[inline]
fn order_key(v: f64) -> u64 {
    let v = if v == 0.0 { 0.0 } else { v };
    let bits = v.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

/// The (value, document) of every document whose value is a number other
/// than NaN, in the value's order, are in two places: most in a sorted list
/// (`settled`), which a range is a slice of, read in order; those written
/// since it was made, in a small B-tree (`recent`). Once that holds an
/// eighth as many, the two are merged into a new list. A document written
/// again or gone leaves its old entry in the list, passed over from then on
/// (`in_settled`), until the next merge drops it. A range read off the list
/// costs a fraction of one read off a B-tree of every value: an entry after
/// the other in memory, not a node after the other.
#[derive(Debug, Clone, Default)]
struct NumericPostings {
    settled: Vec<(u64, u32)>,
    /// The documents whose entry is the one in `settled`.
    in_settled: Bits,
    recent: BTreeSet<(u64, u32)>,
    /// How many documents have a value other than NaN: entries of both.
    ordered: usize,
    /// By document id: its value, if `has` it.
    values: Vec<f64>,
    /// The documents whose field reads as a number, NaN included.
    has: Bits,
}

impl NumericPostings {
    /// The number a value holds, as `filter::eval` reads it.
    fn parse(value: Option<&[u8]>) -> Option<f64> {
        std::str::from_utf8(value?).ok()?.trim().parse::<f64>().ok()
    }

    fn set(&mut self, doc: u32, value: Option<&[u8]>) {
        self.clear(doc);
        let Some(v) = Self::parse(value) else {
            return;
        };
        let at = doc as usize;
        if at >= self.values.len() {
            self.values.resize(at + 1, 0.0);
        }
        self.values[at] = v;
        self.has.set(doc);
        if !v.is_nan() {
            self.recent.insert((order_key(v), doc));
            self.ordered += 1;
            if self.recent.len() > (self.settled.len() / 8).max(1024) {
                self.settle();
            }
        }
    }

    fn clear(&mut self, doc: u32) {
        if !self.has.get(doc) {
            return;
        }
        let v = self.values[doc as usize];
        if !v.is_nan() {
            if self.in_settled.get(doc) {
                self.in_settled.unset(doc);
            } else {
                self.recent.remove(&(order_key(v), doc));
            }
            self.ordered -= 1;
        }
        self.has.unset(doc);
    }

    /// Merge `recent` into a new `settled`, without the entries passed over.
    fn settle(&mut self) {
        let mut recent = std::mem::take(&mut self.recent).into_iter().peekable();
        let mut merged: Vec<(u64, u32)> = Vec::with_capacity(self.ordered);
        for &entry in &self.settled {
            if !self.in_settled.get(entry.1) {
                continue;
            }
            while let Some(&next) = recent.peek() {
                if next < entry {
                    merged.push(next);
                    recent.next();
                } else {
                    break;
                }
            }
            merged.push(entry);
        }
        merged.extend(recent);
        for &(_, doc) in &merged {
            self.in_settled.set(doc);
        }
        self.settled = merged;
    }

    /// Every entry in order, ascending or not, into `each` until it returns
    /// false.
    fn walk(&self, ascending: bool, each: &mut dyn FnMut(u64, u32) -> bool) {
        let live = |entry: &&(u64, u32)| self.in_settled.get(entry.1);
        if ascending {
            merge_walk(
                self.settled.iter().filter(live),
                self.recent.iter(),
                |a, b| a < b,
                each,
            );
        } else {
            merge_walk(
                self.settled.iter().rev().filter(live),
                self.recent.iter().rev(),
                |a, b| a > b,
                each,
            );
        }
    }

    /// The documents whose value is in the range, ends included or not.
    fn range(&self, lo: f64, lo_exclusive: bool, hi: f64, hi_exclusive: bool, out: &mut Bits) {
        if lo.is_nan() || hi.is_nan() {
            return;
        }
        let start = match (order_key(lo), lo_exclusive) {
            (k, false) => (k, 0),
            (u64::MAX, true) => return,
            (k, true) => (k + 1, 0),
        };
        let end = match (order_key(hi), hi_exclusive) {
            (k, false) => (k, u32::MAX),
            (0, true) => return,
            (k, true) => (k - 1, u32::MAX),
        };
        if start > end {
            return;
        }
        let from = self.settled.partition_point(|entry| *entry < start);
        let to = self.settled.partition_point(|entry| *entry <= end);
        for &(_, doc) in &self.settled[from..to] {
            if self.in_settled.get(doc) {
                out.set(doc);
            }
        }
        for &(_, doc) in self.recent.range(start..=end) {
            out.set(doc);
        }
    }

    fn bytes(&self) -> usize {
        // A B-tree node holds up to 11 entries of 16 bytes and two words; at
        // two thirds full, some 26 bytes an entry.
        self.settled.capacity() * 16
            + self.in_settled.bytes()
            + self.recent.len() * 26
            + self.values.capacity() * 8
            + self.has.bytes()
    }
}

/// Two runs of entries, each in order, into `each` in one order, until it
/// returns false: `first(a, b)` says whether `a` comes before `b`.
fn merge_walk<'a>(
    a: impl Iterator<Item = &'a (u64, u32)>,
    b: impl Iterator<Item = &'a (u64, u32)>,
    first: impl Fn(&(u64, u32), &(u64, u32)) -> bool,
    each: &mut dyn FnMut(u64, u32) -> bool,
) {
    let (mut a, mut b) = (a.peekable(), b.peekable());
    loop {
        let next = match (a.peek(), b.peek()) {
            (Some(x), Some(y)) if first(x, y) => a.next(),
            (Some(_), Some(_)) => b.next(),
            (Some(_), None) => a.next(),
            (None, Some(_)) => b.next(),
            (None, None) => return,
        };
        if let Some(&(order, doc)) = next
            && !each(order, doc)
        {
            return;
        }
    }
}

// ── The order of the keys ────────────────────────────────────────────────────
//
// A page ordered by key (FT.SEARCH without SORTBY or words to score) is read
// off the ids in their keys' order, kept here: the first ten of 500,000
// matches are the first ten matches met walking down this list, not the
// least of 500,000 keys compared, each a cache miss away. It is brought up
// to date by the queries that use it, not by writes: an id given out since
// is kept aside (`added`) and compared by its key, and once those are many
// they are merged in. An id that left, or went to another key, keeps its
// old place in `sorted`, which no longer points back at it (`place`), and
// is passed over.

const NO_PLACE: u32 = u32::MAX;

#[derive(Debug, Default)]
struct KeyOrder {
    /// Ids in the order of their keys, as of the last merge.
    sorted: Vec<u32>,
    /// By id: its place in `sorted`, or `NO_PLACE`.
    place: Vec<u32>,
    /// Ids given out since the last merge, in no order. Some have left
    /// since, and some are here twice.
    added: Vec<u32>,
}

/// The key of document `id`, from the postings' table of keys.
#[inline]
fn key_text(keys: &[Option<Arc<str>>], id: u32) -> &str {
    keys.get(id as usize)
        .and_then(|key| key.as_deref())
        .unwrap_or("")
}

/// The first sixteen bytes of `key`, as a number that sorts as they do,
/// shorter keys padded with zeros: two keys whose numbers differ are in that
/// order, and two whose numbers are equal are left to their whole keys.
#[inline]
fn key_prefix(key: &str) -> u128 {
    let mut bytes = [0u8; 16];
    let n = key.len().min(16);
    bytes[..n].copy_from_slice(&key.as_bytes()[..n]);
    u128::from_be_bytes(bytes)
}

impl KeyOrder {
    fn added(&mut self, id: u32) {
        let at = id as usize;
        if at >= self.place.len() {
            self.place.resize(at + 1, NO_PLACE);
        }
        self.place[at] = NO_PLACE;
        self.added.push(id);
    }

    fn removed(&mut self, id: u32) {
        if let Some(place) = self.place.get_mut(id as usize) {
            *place = NO_PLACE;
        }
    }

    #[inline]
    fn placed(&self, id: u32) -> Option<u32> {
        match self.place.get(id as usize) {
            Some(&place) if place != NO_PLACE => Some(place),
            _ => None,
        }
    }

    /// The ids of `added` still waiting for a place, once each.
    fn waiting(&self, keys: &[Option<Arc<str>>]) -> Vec<u32> {
        let mut waiting: Vec<u32> = self
            .added
            .iter()
            .copied()
            .filter(|&id| {
                self.placed(id).is_none() && keys.get(id as usize).is_some_and(Option::is_some)
            })
            .collect();
        waiting.sort_unstable();
        waiting.dedup();
        waiting
    }

    /// Give every id waiting its place: sorted by key, merged with those in
    /// place, every place written again.
    fn merge(&mut self, keys: &[Option<Arc<str>>]) {
        let mut fresh = self.waiting(keys);
        self.added = Vec::new();
        // By the first bytes of the keys, in a list of their own, then by
        // the whole keys where those are equal: sorting by the keys
        // themselves would go to each, at random, at each comparison.
        let mut tagged: Vec<(u128, u32)> = fresh
            .iter()
            .map(|&id| (key_prefix(key_text(keys, id)), id))
            .collect();
        tagged.sort_unstable();
        let mut start = 0;
        while start < tagged.len() {
            let mut end = start + 1;
            while end < tagged.len() && tagged[end].0 == tagged[start].0 {
                end += 1;
            }
            if end - start > 1 {
                tagged[start..end]
                    .sort_unstable_by(|a, b| key_text(keys, a.1).cmp(key_text(keys, b.1)));
            }
            start = end;
        }
        fresh.clear();
        fresh.extend(tagged.iter().map(|&(_, id)| id));

        let mut merged: Vec<u32> = Vec::with_capacity(self.sorted.len() + fresh.len());
        let mut fresh = fresh.into_iter().peekable();
        for (at, &id) in self.sorted.iter().enumerate() {
            if self.place.get(id as usize) != Some(&(at as u32)) {
                continue;
            }
            let key = key_text(keys, id);
            while let Some(&next) = fresh.peek() {
                if key_text(keys, next) < key {
                    merged.push(next);
                    fresh.next();
                } else {
                    break;
                }
            }
            merged.push(id);
        }
        merged.extend(fresh);
        if self.place.len() < keys.len() {
            self.place.resize(keys.len(), NO_PLACE);
        }
        for (at, &id) in merged.iter().enumerate() {
            self.place[id as usize] = at as u32;
        }
        self.sorted = merged;
    }

    fn bytes(&self) -> usize {
        (self.sorted.capacity() + self.place.capacity() + self.added.capacity()) * 4
    }
}

// ── The postings of an index ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Kind {
    Tag(TagPostings),
    Numeric(NumericPostings),
}

#[derive(Debug, Clone)]
struct FieldPostings {
    /// The field it is read from: a hash field's name, a JSON path.
    name: String,
    kind: Kind,
}

/// The TAG and NUMERIC postings of one index, and the document ids they are
/// in terms of. See the top of this file.
#[derive(Debug, Default)]
pub struct Postings {
    /// By id: the document's key.
    keys: Vec<Option<Arc<str>>>,
    ids: AHashMap<Arc<str>, u32>,
    /// Ids given back by documents that left, to be given out again.
    free: Vec<u32>,
    /// Every id in use: every document of the index.
    live: Bits,
    fields: Vec<FieldPostings>,
    /// Queries do not use the postings (`FT.DEBUG POSTINGS index OFF`): every
    /// document is tried, as before there were any. For comparing the two.
    /// They are still kept.
    pub unused: bool,
    /// Queries read every document they match, not only those of the page
    /// they return (`FT.DEBUG PAGE index OFF`): see `search::page_from_postings`.
    /// For comparing the two.
    pub read_all: bool,
    /// The ids in the order of their keys: see `KeyOrder`. Behind a lock of
    /// its own, as queries bring it up to date under a read lock of these:
    /// they read it together, and one at a time merges into it.
    order: parking_lot::RwLock<KeyOrder>,
    /// By id: its key's cluster hash slot, for a query in a cluster to keep
    /// only the documents of the slots its node serves without hashing
    /// every key it matches.
    slots: Vec<u16>,
}

/// The postings of an index, shared by its clones.
pub type SharedPostings = Arc<parking_lot::RwLock<Postings>>;

impl Postings {
    /// Postings for the TAG and NUMERIC fields of `schema`.
    pub fn new(schema: &[SchemaField]) -> Self {
        let mut postings = Self::default();
        postings.add_fields(schema);
        postings
    }

    /// Start keeping postings for the TAG and NUMERIC fields of `fields`
    /// (`FT.ALTER`). They are empty: the caller fills them in.
    pub fn add_fields(&mut self, fields: &[SchemaField]) {
        for field in fields {
            let kind = match field.field_type {
                FieldType::Tag => Kind::Tag(TagPostings::new(
                    field.options.separator,
                    field.options.case_sensitive,
                )),
                FieldType::Numeric => Kind::Numeric(NumericPostings::default()),
                _ => continue,
            };
            self.fields.push(FieldPostings {
                name: field.name.clone(),
                kind,
            });
        }
    }

    /// Whether this index keeps postings at all: it has a TAG or a NUMERIC
    /// field.
    #[inline]
    pub fn enabled(&self) -> bool {
        !self.fields.is_empty()
    }

    /// Whether a write of the field `name` is one these postings keep.
    pub fn reads(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name)
    }

    pub fn id(&self, key: &str) -> Option<u32> {
        self.ids.get(key).copied()
    }

    pub fn key(&self, id: u32) -> Option<&Arc<str>> {
        self.keys.get(id as usize).and_then(Option::as_ref)
    }

    /// The cluster hash slot of document `id`'s key.
    #[inline]
    pub fn slot(&self, id: u32) -> u16 {
        self.slots.get(id as usize).copied().unwrap_or(0)
    }

    /// Every document of the index.
    pub fn live(&self) -> &Bits {
        &self.live
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The document at `key`'s id, given it one if it has none.
    pub fn ensure(&mut self, key: &str) -> u32 {
        if let Some(&id) = self.ids.get(key) {
            return id;
        }
        let key: Arc<str> = Arc::from(key);
        let id = match self.free.pop() {
            Some(id) => {
                self.keys[id as usize] = Some(Arc::clone(&key));
                id
            }
            None => {
                self.keys.push(Some(Arc::clone(&key)));
                (self.keys.len() - 1) as u32
            }
        };
        let at = id as usize;
        if at >= self.slots.len() {
            self.slots.resize(at + 1, 0);
        }
        self.slots[at] = crate::cluster::key_hash_slot(key.as_bytes());
        self.ids.insert(key, id);
        self.live.set(id);
        let order = self.order.get_mut();
        order.added(id);
        // Never queried by key, the list of ids waiting would only grow.
        if order.added.len() > 2 * self.ids.len() + 4096 {
            order.added = order.waiting(&self.keys);
        }
        id
    }

    /// Record what the document at `key` holds now, from `fields`, which are
    /// the fields that were written. With `whole`, they are all it has: a
    /// field kept here that is not among them is one it no longer holds.
    /// Returns its id.
    pub fn write(&mut self, key: &str, fields: &[(String, Arc<[u8]>)], whole: bool) -> u32 {
        let id = self.ensure(key);
        let slots = self.keys.len();
        for field in &mut self.fields {
            let value = fields
                .iter()
                .find(|(n, _)| *n == field.name)
                .map(|(_, v)| &**v);
            if value.is_none() && !whole {
                continue;
            }
            match &mut field.kind {
                Kind::Tag(tags) => tags.set(id, value, slots),
                Kind::Numeric(numbers) => numbers.set(id, value),
            }
        }
        id
    }

    /// [`Postings::write`] of a document's fields as an index reads them.
    pub fn write_doc(&mut self, key: &str, doc: &super::Fields) -> u32 {
        let id = self.ensure(key);
        let slots = self.keys.len();
        for field in &mut self.fields {
            let value = doc.get(&field.name).map(|v| &**v);
            match &mut field.kind {
                Kind::Tag(tags) => tags.set(id, value, slots),
                Kind::Numeric(numbers) => numbers.set(id, value),
            }
        }
        id
    }

    /// The document at `key` has left the index: out of every set, and its
    /// id is given back.
    pub fn remove(&mut self, key: &str) {
        let Some(id) = self.ids.remove(key) else {
            return;
        };
        let slots = self.keys.len();
        for field in &mut self.fields {
            match &mut field.kind {
                Kind::Tag(tags) => tags.clear(id, slots),
                Kind::Numeric(numbers) => numbers.clear(id),
            }
        }
        self.keys[id as usize] = None;
        self.live.unset(id);
        self.free.push(id);
        self.order.get_mut().removed(id);
    }

    /// Forget every document (the index is being filled again).
    pub fn clear(&mut self) {
        let mut fresh = Postings {
            unused: self.unused,
            read_all: self.read_all,
            ..Postings::default()
        };
        for field in &self.fields {
            fresh.fields.push(FieldPostings {
                name: field.name.clone(),
                kind: match &field.kind {
                    Kind::Tag(t) => Kind::Tag(TagPostings::new(t.separator, t.case_sensitive)),
                    Kind::Numeric(_) => Kind::Numeric(NumericPostings::default()),
                },
            });
        }
        *self = fresh;
    }

    /// The documents a `@field:{…}` clause matches, if this field's postings
    /// are kept with the clause's separator and case: `None` otherwise.
    pub fn tag(
        &self,
        field: &str,
        values: &[String],
        prefixes: &[String],
        separator: char,
        case_sensitive: bool,
    ) -> Option<Bits> {
        let tags = self.fields.iter().find_map(|f| match &f.kind {
            Kind::Tag(t)
                if f.name == field
                    && t.separator == separator
                    && t.case_sensitive == case_sensitive =>
            {
                Some(t)
            }
            _ => None,
        })?;
        let mut out = Bits::new();
        tags.any_of(values, prefixes, &mut out);
        Some(out)
    }

    fn numeric_field(&self, field: &str) -> Option<&NumericPostings> {
        self.fields.iter().find_map(|f| match &f.kind {
            Kind::Numeric(n) if f.name == field => Some(n),
            _ => None,
        })
    }

    /// The first `need` of `ids` (`of` of them) in the order of their keys.
    /// `ran_out` is asked at each step of the walk that finds them, and once
    /// it says so they are the first of those found.
    pub fn first_by_key(
        &self,
        ids: &Bits,
        need: usize,
        of: usize,
        ran_out: &mut dyn FnMut() -> bool,
    ) -> Vec<u32> {
        if need == 0 {
            return Vec::new();
        }
        // A sixteenth more, and at least a few hundred: a merge goes through
        // every place, and the waiting are sorted again at each query.
        let behind = |order: &KeyOrder| order.added.len() > (order.sorted.len() / 16).max(256);
        let mut order = self.order.read();
        if behind(&order) {
            drop(order);
            let mut merging = self.order.write();
            // Unless another query has merged meanwhile.
            if behind(&merging) {
                merging.merge(&self.keys);
            }
            order = parking_lot::RwLockWriteGuard::downgrade(merging);
        }
        let order = &*order;
        // Those with a place. Walking the places meets the `need` first
        // after about need * places / of of them; looking up the place of
        // each, after `of`. Whichever is fewer.
        let mut first: Vec<u32> = Vec::with_capacity(need.min(of));
        let places = order.sorted.len();
        if (need as u128) * (places as u128) < (of as u128) * (of as u128) {
            for (at, &id) in order.sorted.iter().enumerate() {
                if ran_out() {
                    break;
                }
                if order.place[id as usize] == at as u32 && ids.get(id) {
                    first.push(id);
                    if first.len() == need {
                        break;
                    }
                }
            }
        } else {
            let mut placed: Vec<(u32, u32)> = Vec::with_capacity(of);
            for id in ids.iter() {
                if ran_out() {
                    break;
                }
                if let Some(place) = order.placed(id) {
                    placed.push((place, id));
                }
            }
            if need < placed.len() {
                placed.select_nth_unstable(need);
                placed.truncate(need);
            }
            placed.sort_unstable();
            first.extend(placed.iter().map(|&(_, id)| id));
        }
        // Those still waiting for a place: by their keys, then the two lists
        // merged.
        let mut late: Vec<u32> = order
            .added
            .iter()
            .copied()
            .filter(|&id| ids.get(id) && order.placed(id).is_none())
            .collect();
        if late.is_empty() {
            return first;
        }
        late.sort_unstable();
        late.dedup();
        let key = |id: u32| key_text(&self.keys, id);
        if need < late.len() {
            late.select_nth_unstable_by(need, |&a, &b| key(a).cmp(key(b)));
            late.truncate(need);
        }
        late.sort_unstable_by(|&a, &b| key(a).cmp(key(b)));
        let mut out = Vec::with_capacity(need);
        let (mut a, mut b) = (first.into_iter().peekable(), late.into_iter().peekable());
        while out.len() < need {
            let next = match (a.peek(), b.peek()) {
                (Some(&x), Some(&y)) if key(x) < key(y) => a.next(),
                (Some(_), Some(_)) => b.next(),
                (Some(_), None) => a.next(),
                (None, Some(_)) => b.next(),
                (None, None) => break,
            };
            out.extend(next);
        }
        out
    }

    /// The postings of the NUMERIC field read from `field`, to order by.
    pub fn numbers(&self, field: &str) -> Option<Numbers<'_>> {
        self.numeric_field(field).map(Numbers)
    }

    /// The documents a `@field:[lo hi]` clause matches.
    pub fn range(
        &self,
        field: &str,
        lo: f64,
        lo_exclusive: bool,
        hi: f64,
        hi_exclusive: bool,
    ) -> Option<Bits> {
        let numbers = self.numeric_field(field)?;
        let mut out = Bits::new();
        numbers.range(lo, lo_exclusive, hi, hi_exclusive, &mut out);
        Some(out)
    }

    /// The documents a `@field != value` clause matches: those with a number
    /// there (NaN too, which equals nothing) but not `value`.
    pub fn not_equal(&self, field: &str, value: f64) -> Option<Bits> {
        let numbers = self.numeric_field(field)?;
        let mut out = numbers.has.clone();
        let mut equal = Bits::new();
        numbers.range(value, false, value, false, &mut equal);
        out.and_not(&equal);
        Some(out)
    }

    /// Bytes held: the id tables, and each field's postings.
    pub fn bytes(&self) -> PostingsBytes {
        let ids = self.keys.capacity() * 16
            + self.ids.capacity() * 24
            + self
                .keys
                .iter()
                .flatten()
                .map(|k| 16 + k.len())
                .sum::<usize>()
            + self.free.capacity() * 4
            + self.slots.capacity() * 2
            + self.live.bytes()
            + self.order.read().bytes();
        let mut tags = 0;
        let mut numbers = 0;
        let mut tag_values = 0;
        for field in &self.fields {
            match &field.kind {
                Kind::Tag(t) => {
                    tags += t.bytes();
                    tag_values += t.numbers.len();
                }
                Kind::Numeric(n) => numbers += n.bytes(),
            }
        }
        PostingsBytes {
            ids,
            tags,
            numbers,
            tag_values,
        }
    }
}

/// The postings of one NUMERIC field, as `SORTBY` reads them.
pub struct Numbers<'a>(&'a NumericPostings);

impl<'a> Numbers<'a> {
    /// The documents whose field reads as a number, NaN included.
    pub fn has(&self) -> &'a Bits {
        &self.0.has
    }

    /// Whether some document's value is NaN: those are in no order here.
    pub fn any_nan(&self) -> bool {
        self.0.has.count() != self.0.ordered
    }

    /// How many documents have a value other than NaN: what [`Numbers::walk`]
    /// goes through.
    pub fn ordered(&self) -> usize {
        self.0.ordered
    }

    /// The number document `id` holds, if it holds one.
    #[inline]
    pub fn value(&self, id: u32) -> Option<f64> {
        if self.0.has.get(id) {
            Some(self.0.values[id as usize])
        } else {
            None
        }
    }

    /// Every document with a value other than NaN, in the order of its
    /// value, ascending or descending, into `each` until it returns false:
    /// each with the value's place in that order, which is the same for
    /// equal values (minus zero and zero too). Equal values come in the
    /// order of their ids, or the reverse.
    pub fn walk(&self, ascending: bool, mut each: impl FnMut(u64, u32) -> bool) {
        self.0.walk(ascending, &mut each);
    }
}

/// What [`Postings::bytes`] counts, for `FT.INFO`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PostingsBytes {
    pub ids: usize,
    pub tags: usize,
    pub numbers: usize,
    /// Distinct tags, over all TAG fields.
    pub tag_values: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::FieldOptions;

    fn schema() -> Vec<SchemaField> {
        vec![
            SchemaField {
                name: "t".into(),
                field_type: FieldType::Tag,
                options: FieldOptions::default(),
            },
            SchemaField {
                name: "n".into(),
                field_type: FieldType::Numeric,
                options: FieldOptions::default(),
            },
        ]
    }

    fn put(p: &mut Postings, key: &str, t: &str, n: &str) -> u32 {
        p.write(
            key,
            &[
                ("t".to_string(), Arc::from(t.as_bytes())),
                ("n".to_string(), Arc::from(n.as_bytes())),
            ],
            true,
        )
    }

    fn keys(p: &Postings, bits: &Bits) -> Vec<String> {
        let mut out: Vec<String> = bits
            .iter()
            .map(|id| p.key(id).unwrap().to_string())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn bits_set_count_iterate_and_combine() {
        let mut a = Bits::new();
        for id in [0, 63, 64, 1000] {
            a.set(id);
        }
        assert_eq!(a.count(), 4);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![0, 63, 64, 1000]);
        let mut b = Bits::new();
        b.set(64);
        b.set(5);
        let mut and = a.clone();
        and.and_with(&b);
        assert_eq!(and.iter().collect::<Vec<_>>(), vec![64]);
        let mut or = b.clone();
        or.or_with(&a);
        assert_eq!(or.count(), 5);
        a.and_not(&b);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![0, 63, 1000]);
        a.unset(1000);
        assert!(!a.get(1000));
    }

    #[test]
    fn tags_are_read_as_the_filter_reads_them() {
        let mut p = Postings::new(&schema());
        put(&mut p, "a", " Red , blue ", "1");
        put(&mut p, "b", "red", "2");
        put(&mut p, "c", "green", "3");
        let red = p.tag("t", &["red".into()], &[], ',', false).unwrap();
        assert_eq!(keys(&p, &red), ["a", "b"]);
        let pre = p
            .tag("t", &[], &["bl".into(), "gr".into()], ',', false)
            .unwrap();
        assert_eq!(keys(&p, &pre), ["a", "c"]);
        // Another separator is another way of reading: not these postings.
        assert!(p.tag("t", &["red".into()], &[], ';', false).is_none());
    }

    #[test]
    fn a_document_written_again_or_gone_leaves_its_old_sets() {
        let mut p = Postings::new(&schema());
        put(&mut p, "a", "red", "1");
        put(&mut p, "a", "blue", "5");
        assert!(
            p.tag("t", &["red".into()], &[], ',', false)
                .unwrap()
                .is_empty()
        );
        assert_eq!(p.range("n", 0.0, false, 2.0, false).unwrap().count(), 0);
        assert_eq!(p.range("n", 5.0, false, 5.0, false).unwrap().count(), 1);
        // A write of other fields leaves these alone.
        p.write("a", &[("x".to_string(), Arc::from(&b"y"[..]))], false);
        assert_eq!(
            p.tag("t", &["blue".into()], &[], ',', false)
                .unwrap()
                .count(),
            1
        );
        p.remove("a");
        assert!(
            p.tag("t", &["blue".into()], &[], ',', false)
                .unwrap()
                .is_empty()
        );
        assert!(p.live().is_empty());
        assert_eq!(p.slot(0), crate::cluster::key_hash_slot(b"a"));
        // Its id is given out again.
        assert_eq!(put(&mut p, "b", "red", "1"), 0);
    }

    #[test]
    fn ranges_and_their_ends() {
        let mut p = Postings::new(&schema());
        for (key, n) in [
            ("a", "-inf"),
            ("b", "-0"),
            ("c", "0"),
            ("d", "1.5"),
            ("e", "nan"),
            ("f", "x"),
            ("g", "inf"),
        ] {
            put(&mut p, key, "", n);
        }
        let r = |lo, lx, hi, hx| keys(&p, &p.range("n", lo, lx, hi, hx).unwrap());
        assert_eq!(r(0.0, false, 0.0, false), ["b", "c"]);
        assert_eq!(r(-0.0, true, f64::INFINITY, false), ["d", "g"]);
        assert_eq!(r(f64::NEG_INFINITY, false, 0.0, true), ["a"]);
        assert_eq!(
            r(f64::NEG_INFINITY, false, f64::INFINITY, false),
            ["a", "b", "c", "d", "g"]
        );
        assert!(r(2.0, false, 1.0, false).is_empty());
        assert!(r(1.5, true, 1.5, true).is_empty());
        assert!(r(f64::NAN, false, 1.0, false).is_empty());
        // NaN differs from everything; a value that is no number is not there.
        assert_eq!(
            keys(&p, &p.not_equal("n", 0.0).unwrap()),
            ["a", "d", "e", "g"]
        );
    }

    #[test]
    fn keys_come_in_their_order_through_writes_and_merges() {
        let mut p = Postings::new(&schema());
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut ran_out = || false;
        for round in 0..30 {
            // Keys added, removed and added again: ids go to other keys, and
            // the waiting list grows past what makes a query merge it.
            for _ in 0..700 {
                let key = format!("k{}", next() % 6000);
                if next() % 3 == 0 {
                    p.remove(&key);
                } else {
                    put(&mut p, &key, "", "1");
                }
            }
            let keep = [100, 60, 20, 3][round % 4];
            let mut ids = Bits::new();
            for id in p.live().iter() {
                if next() % 100 < keep {
                    ids.set(id);
                }
            }
            let of = ids.count();
            let mut whole: Vec<u32> = ids.iter().collect();
            whole.sort_by(|&a, &b| p.key(a).unwrap().cmp(p.key(b).unwrap()));
            for need in [1, 2, 10, of / 3, of] {
                let need = need.min(of);
                let first = p.first_by_key(&ids, need, of, &mut ran_out);
                assert_eq!(
                    first.as_slice(),
                    &whole[..need],
                    "round {round}, {need} of {of}"
                );
            }
        }
        // Keys alike in their first sixteen bytes, and one a prefix of
        // another.
        let mut p = Postings::new(&schema());
        for key in [
            "same-sixteen-byteZ",
            "same-sixteen-byte",
            "same-sixteen-byteA",
            "s",
            "",
        ] {
            put(&mut p, key, "", "1");
        }
        p.order.write().merge(&p.keys);
        let all = p.live().clone();
        let first: Vec<String> = p
            .first_by_key(&all, 5, 5, &mut ran_out)
            .into_iter()
            .map(|id| p.key(id).unwrap().to_string())
            .collect();
        assert_eq!(
            first,
            [
                "",
                "s",
                "same-sixteen-byte",
                "same-sixteen-byteA",
                "same-sixteen-byteZ"
            ]
        );
    }

    #[test]
    fn numbers_settled_and_recent_answer_as_one() {
        // Enough writes for several merges, values written again and gone
        // between them, every range and both walks checked against the
        // values themselves.
        let mut p = Postings::new(&schema());
        let mut seed = 0x1319_8a2e_0370_7344u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut model: std::collections::BTreeMap<String, f64> = Default::default();
        for round in 0..12 {
            for _ in 0..1500 {
                let key = format!("k{}", next() % 3000);
                match next() % 5 {
                    0 => {
                        p.remove(&key);
                        model.remove(&key);
                    }
                    1 => {
                        // No number: out of the field.
                        put(&mut p, &key, "", "x");
                        model.remove(&key);
                    }
                    _ => {
                        let v = (next() % 200) as f64 - 100.0;
                        put(&mut p, &key, "", &v.to_string());
                        model.insert(key, v);
                    }
                }
            }
            let numbers = p.numbers("n").unwrap();
            assert_eq!(numbers.ordered(), model.len(), "round {round}");
            assert!(!numbers.any_nan());
            for _ in 0..20 {
                let a = (next() % 220) as f64 - 110.0;
                let b = (next() % 220) as f64 - 110.0;
                let (lo, hi) = (a.min(b), a.max(b));
                let (lx, hx) = (next() % 2 == 0, next() % 2 == 0);
                let mut want: Vec<String> = model
                    .iter()
                    .filter(|&(_, &v)| {
                        (if lx { v > lo } else { v >= lo }) && (if hx { v < hi } else { v <= hi })
                    })
                    .map(|(k, _)| k.clone())
                    .collect();
                want.sort();
                let got = keys(&p, &p.range("n", lo, lx, hi, hx).unwrap());
                assert_eq!(got, want, "round {round}: {lo} {lx} {hi} {hx}");
            }
            for ascending in [true, false] {
                let mut walked: Vec<(f64, u32)> = Vec::new();
                numbers.walk(ascending, |_, id| {
                    walked.push((numbers.value(id).unwrap(), id));
                    true
                });
                assert_eq!(walked.len(), model.len());
                let mut ids: Vec<u32> = walked.iter().map(|w| w.1).collect();
                ids.sort_unstable();
                ids.dedup();
                assert_eq!(ids.len(), walked.len(), "an id twice");
                assert!(walked.windows(2).all(|w| if ascending {
                    (w[0].0, w[0].1) < (w[1].0, w[1].1)
                } else {
                    (w[0].0, w[0].1) > (w[1].0, w[1].1)
                }));
            }
        }
    }

    #[test]
    fn numbers_are_walked_in_their_order_either_way() {
        let mut p = Postings::new(&schema());
        for (key, n) in [("a", "3"), ("b", "-1"), ("c", "3"), ("d", "x"), ("e", "-0")] {
            put(&mut p, key, "", n);
        }
        let numbers = p.numbers("n").unwrap();
        assert!(!numbers.any_nan());
        assert_eq!(numbers.ordered(), 4);
        assert_eq!(numbers.has().count(), 4);
        assert_eq!(numbers.value(p.id("d").unwrap()), None);
        assert_eq!(numbers.value(p.id("b").unwrap()), Some(-1.0));
        let walked = |ascending: bool| {
            let mut out = Vec::new();
            numbers.walk(ascending, |_, id| {
                out.push(p.key(id).unwrap().to_string());
                true
            });
            out
        };
        assert_eq!(walked(true), ["b", "e", "a", "c"]);
        assert_eq!(walked(false), ["c", "a", "e", "b"]);
        // Stopped by its callback.
        let mut seen = 0;
        numbers.walk(true, |_, _| {
            seen += 1;
            seen < 2
        });
        assert_eq!(seen, 2);
        put(&mut p, "f", "", "nan");
        assert!(p.numbers("n").unwrap().any_nan());
        assert!(p.numbers("t").is_none());
    }

    #[test]
    fn a_large_tag_set_becomes_a_bitmap_and_back() {
        let mut p = Postings::new(&schema());
        for i in 0..1000 {
            put(
                &mut p,
                &format!("k{i}"),
                if i % 2 == 0 { "even" } else { "odd" },
                "1",
            );
        }
        let even = p.tag("t", &["even".into()], &[], ',', false).unwrap();
        assert_eq!(even.count(), 500);
        for i in (0..1000).step_by(2).skip(5) {
            p.remove(&format!("k{i}"));
        }
        let even = p.tag("t", &["even".into()], &[], ',', false).unwrap();
        assert_eq!(keys(&p, &even), ["k0", "k2", "k4", "k6", "k8"]);
    }
}
