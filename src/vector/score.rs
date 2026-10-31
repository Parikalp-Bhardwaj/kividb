// Relevance: what `WITHSCORES`, `ADDSCORES` and the order of FT.SEARCH are
// in terms of, computed as RediSearch (Redis 8's query engine) computes it.
//
// A query is a tree, and a document's score is the score of the part of the
// tree it matched: a word scores by its frequency in the document and its
// rarity in the index, and a union or an intersection is the sum of what
// its children that matched score. What RediSearch does, and this does too:
//
// - A word's frequency is over the whole document, every TEXT field of it,
//   and an occurrence counts the field's `WEIGHT` (truncated, at least 1),
//   not 1: a `WEIGHT 2` field makes a word count twice. The document's
//   length is its words counted the same way.
// - A stem (`+prefer`, see `stem`) is a term of its own, with its own
//   document frequency, and an occurrence of it counts 0.2 of the field's
//   weight, at least 1. A query's word that has a stem looks for both, so
//   a document that holds the very word scores for the word and for its
//   stem.
// - A TAG value the query asks for is a term too, of frequency 1, scored
//   against the document's (TEXT) length. A NUMERIC clause, `NOT`, and an
//   optional clause that did not match score nothing. `*` scores as a word
//   of frequency 1 and IDF 1 would.
// - A prefix, suffix, infix or fuzzy term scores one of the document's
//   words it matches, not all of them.
//
// Known differences: which word that is (here, the first in order; in
// RediSearch, it depends on how far its iterators have read); a phrase in
// RediSearch also matches across the end of one field and the start of the
// next; and for TFIDF, TFIDF.DOCNORM and BM25 with a NUMERIC clause,
// RediSearch's slop (see [`slop`]) can come out of an overflow on the
// numeric range's positions, which depends on how its range tree is split.
// - The corpus is every document of the index, including one with no TEXT
//   field: its size is the N of the IDF, and the average length is over it.
//
// The scorers are those of `SCORER`: BM25STD (the default), BM25STD.TANH,
// BM25 (RediSearch's older variant), TFIDF, TFIDF.DOCNORM, DISMAX, DOCSCORE
// and HAMMING (no payloads here: always 0).

use super::filter::{Doc, Filter, TextMatch};
use super::stem::{STEM_PREFIX, is_stem};
use super::{FieldType, IndexDef};
use ahash::AHashMap;

/// A ranking function of `SCORER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scorer {
    #[default]
    Bm25Std,
    Bm25StdTanh,
    Bm25,
    Tfidf,
    TfidfDocNorm,
    DisMax,
    DocScore,
    Hamming,
}

impl Scorer {
    pub fn parse(name: &str) -> Option<Scorer> {
        Some(match name.to_ascii_uppercase().as_str() {
            "BM25STD" => Scorer::Bm25Std,
            "BM25STD.TANH" => Scorer::Bm25StdTanh,
            "BM25" => Scorer::Bm25,
            "TFIDF" => Scorer::Tfidf,
            "TFIDF.DOCNORM" => Scorer::TfidfDocNorm,
            "DISMAX" => Scorer::DisMax,
            "DOCSCORE" => Scorer::DocScore,
            "HAMMING" => Scorer::Hamming,
            _ => return None,
        })
    }

    /// `FT.CONFIG DEFAULT_SCORER`, or BM25STD if it names none.
    pub fn default_scorer() -> Scorer {
        Scorer::parse(&crate::commands::search_tools::config_value(
            "DEFAULT_SCORER",
        ))
        .unwrap_or_default()
    }
}

/// BM25STD's term weight as RediSearch's C computes it, to the last digit:
/// `k1` and `b` are `float`s there, and so are `k1 + 1` and `b * docLen`.
fn bm25std(idf: f64, f: f64, len: u32, avgdl: f64) -> f64 {
    let k1 = 1.2f32;
    let b = 0.75f32;
    let norm = if avgdl > 0.0 {
        f64::from(b * len as f32) / avgdl
    } else {
        0.0
    };
    idf * f * f64::from(k1 + 1.0) / (f + f64::from(k1) * (f64::from(1.0f32 - b) + norm))
}
/// RediSearch's older BM25's saturation, `f / (f + k1 (1 - b + b avgdl))`
/// with `b` 0.5 and no document length, in its `float`s.
fn bm25_legacy(f: f64, avgdl: f64) -> f64 {
    let k1 = 1.2f32;
    let b = 0.5f32;
    f / (f + f64::from(k1) * (f64::from(1.0f32 - b) + f64::from(b) * avgdl))
}

/// What an occurrence of a stem counts, of the field's weight.
const STEM_FACTOR: f32 = 0.2;

/// `ln(1 + (N - df + 0.5) / (df + 0.5))`, as RediSearch's BM25STD has it.
fn idf_bm25(n: usize, df: usize) -> f64 {
    let n = n.max(df) as f64;
    let df = df as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// `floor(log2(1 + (N + 1) / df))`, RediSearch's IDF of TFIDF and BM25.
fn idf_tfidf(n: usize, df: usize) -> f64 {
    let df = df.max(1) as f64;
    (1.0 + (n + 1) as f64 / df).log2().floor()
}

/// How often each term is in one document, where, and in which fields.
#[derive(Default)]
struct Stat {
    freq: u32,
    /// Bit `i`: the `i`th TEXT field of the schema holds it.
    fields: u64,
    offsets: Vec<u32>,
}

/// One document's terms, as the index has them (its forward index).
#[derive(Default)]
struct DocTerms<'a> {
    /// Its words, each counted by its field's weight.
    len: u32,
    max_freq: u32,
    terms: AHashMap<&'a str, Stat>,
}

/// The bit of TEXT field `name` (see [`Stat::fields`]).
fn field_bit(def: &IndexDef, name: &str) -> u64 {
    def.fields
        .iter()
        .filter(|f| matches!(f.field_type, FieldType::Text) && !f.options.noindex)
        .position(|f| f.name == name)
        .map_or(0, |i| 1u64 << i.min(63))
}

fn fields_mask(def: &IndexDef, fields: &[String]) -> u64 {
    fields.iter().fold(0, |mask, f| mask | field_bit(def, f))
}

impl<'a> DocTerms<'a> {
    /// The document's length, and the terms of `wanted` it holds; every term
    /// if `wanted` is `None`.
    fn read(
        def: &IndexDef,
        fields: &'a AHashMap<String, Vec<String>>,
        wanted: Option<&ahash::AHashSet<String>>,
    ) -> Self {
        let mut doc = DocTerms::default();
        let mut pos = 0u32;
        let text = def
            .fields
            .iter()
            .filter(|f| matches!(f.field_type, FieldType::Text) && !f.options.noindex);
        for (i, field) in text.enumerate() {
            let Some(tokens) = fields.get(&field.name) else {
                continue;
            };
            let weight = field.options.weight as f32;
            let unit = (weight as u32).max(1);
            let stem_unit = ((weight * STEM_FACTOR) as u32).max(1);
            let bit = 1u64 << i.min(63);
            for token in tokens {
                let freq = if is_stem(token) {
                    stem_unit
                } else {
                    pos += 1;
                    doc.len += unit;
                    unit
                };
                if wanted.is_some_and(|w| !w.contains(token.as_str())) {
                    continue;
                }
                let stat = doc.terms.entry(token.as_str()).or_default();
                stat.freq += freq;
                stat.fields |= bit;
                stat.offsets.push(pos);
            }
        }
        doc.max_freq = doc.terms.values().map(|s| s.freq).max().unwrap_or(0);
        doc
    }
}

/// What a result is, as RediSearch's kinds of results go: it matters to how
/// it is scored, and to whether it has positions (see [`Res::has_offsets`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Term,
    /// `*`, `NOT`, an optional clause.
    Virtual,
    Numeric,
    Union,
    Intersection,
}

/// The part of a query one document matched.
struct Res {
    kind: Kind,
    value: f64,
    /// Term: where in the document; an aggregate: its children's, merged.
    offsets: Vec<u32>,
    children: Vec<Res>,
}

impl Res {
    fn leaf(kind: Kind, value: f64, offsets: Vec<u32>) -> Res {
        Res {
            kind,
            value,
            offsets,
            children: Vec::new(),
        }
    }

    fn has_offsets(&self) -> bool {
        match self.kind {
            Kind::Term => !self.offsets.is_empty(),
            Kind::Union | Kind::Intersection => {
                let mut kinds = self.children.iter().map(|c| c.kind);
                let first = kinds.next();
                let same = |k: Kind| first == Some(k) && self.children.iter().all(|c| c.kind == k);
                !(same(Kind::Virtual) || same(Kind::Numeric))
            }
            Kind::Virtual | Kind::Numeric => false,
        }
    }
}

/// Whether `filter` is a query word with its stems (`word | +stem | stem`,
/// see `filter::Parser::exact`): one union of its own, which a union it is in
/// does not take apart.
pub fn is_expansion(filter: &Filter) -> bool {
    match filter {
        Filter::Or(_, right) => matches!(
            &**right,
            Filter::Text {
                term: TextMatch::Stem(..) | TextMatch::Expanded(..),
                ..
            }
        ),
        _ => false,
    }
}

/// The operands of a run of one operator, as RediSearch has them: `a b c`
/// is one intersection of three, `a | b | c` one union of three, and a word
/// with its stems is one union.
///
/// A run is folded to the left (`a b c` is `(a b) c`), and RediSearch's
/// parser takes a group apart only where it starts the run: `(a b) c` is
/// one intersection of three, `a (b c)` an intersection of `a` and another.
/// (It also keeps `@t:(a b) c`'s group whole, which this cannot tell from
/// `@t:a @t:b c`.)
pub fn operands<'f>(filter: &'f Filter, out: &mut Vec<&'f Filter>) {
    fn walk<'f>(filter: &'f Filter, and: bool, inside: bool, out: &mut Vec<&'f Filter>) {
        match filter {
            Filter::And(l, r) if and => {
                walk(l, and, inside, out);
                out.push(r);
            }
            Filter::Or(l, r) if !and && (inside || !is_expansion(filter)) => {
                walk(l, and, inside, out);
                // A word's stems are flat in their union.
                if inside {
                    walk(r, and, inside, out);
                } else {
                    out.push(r);
                }
            }
            other => out.push(other),
        }
    }
    match filter {
        Filter::And(..) => walk(filter, true, false, out),
        Filter::Or(..) => walk(filter, false, is_expansion(filter), out),
        other => out.push(other),
    }
}

/// A query's scoring, with what it needs of the index read once.
pub struct Scoring<'q> {
    filter: &'q Filter,
    scorer: Scorer,
    /// Documents in the index.
    n: usize,
    avgdl: f64,
    /// Document frequencies of the words and stems the query names.
    term_df: AHashMap<String, usize>,
    /// Of the TAG values it names, by field.
    tag_df: AHashMap<(String, String), usize>,
    /// The terms a document's are read for: `None` for all of them (a
    /// pattern matches terms not known in advance, and TFIDF divides by the
    /// frequency of the document's most frequent term).
    wanted: Option<ahash::AHashSet<String>>,
    tanh_factor: f64,
}

impl<'q> Scoring<'q> {
    pub fn new(def: &IndexDef, filter: &'q Filter, scorer: Scorer) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        let mut words = Vec::new();
        let mut tags = Vec::new();
        let patterns = named_terms(filter, &mut words, &mut tags);
        let wanted =
            (!patterns && scorer != Scorer::Tfidf).then(|| words.iter().cloned().collect());
        let term_df = {
            let df = def.doc_freq.read();
            words
                .into_iter()
                .map(|w| {
                    let n = df.get(&w).copied().unwrap_or(0);
                    (w, n)
                })
                .collect()
        };
        let (n, tag_df) = {
            let postings = def.postings.read();
            let n = if postings.enabled() {
                postings.len()
            } else {
                def.doc_count.load(Relaxed)
            };
            let tag_df = tags
                .into_iter()
                .map(|(field, value)| {
                    let n = postings.tag_docs(&field, &value);
                    ((field, value), n)
                })
                .collect();
            (n, tag_df)
        };
        let n = n.max(def.doc_count.load(Relaxed));
        let avgdl = if n == 0 {
            0.0
        } else {
            def.total_tokens.load(Relaxed) as f64 / n as f64
        };
        let tanh_factor =
            crate::commands::search_tools::config_number("BM25STD_TANH_FACTOR", 4).max(1) as f64;
        Scoring {
            filter,
            scorer,
            n,
            avgdl,
            term_df,
            tag_df,
            wanted,
            tanh_factor,
        }
    }

    /// The score of the document at `key`, whose fields are `doc`. The
    /// document matches the query.
    pub fn score(&self, def: &IndexDef, key: &str, doc: &dyn Doc) -> f64 {
        match self.scorer {
            Scorer::DocScore => return 1.0,
            Scorer::Hamming => return 0.0,
            _ => {}
        }
        let forward = def.forward.read();
        let empty = AHashMap::new();
        let terms = DocTerms::read(
            def,
            forward.get(key).unwrap_or(&empty),
            self.wanted.as_ref(),
        );
        let Some(res) = self.node(def, self.filter, &terms, doc) else {
            return 0.0;
        };
        let raw = res.value;
        match self.scorer {
            Scorer::Bm25Std => raw,
            Scorer::Bm25StdTanh => (raw / self.tanh_factor).tanh(),
            Scorer::Bm25 => raw / f64::from(slop(&res)),
            Scorer::Tfidf | Scorer::TfidfDocNorm => {
                let norm = if self.scorer == Scorer::Tfidf {
                    terms.max_freq
                } else {
                    terms.len
                };
                if norm == 0 {
                    return 0.0;
                }
                raw / f64::from(norm) / f64::from(slop(&res))
            }
            Scorer::DisMax => raw,
            Scorer::DocScore | Scorer::Hamming => unreachable!(),
        }
    }

    /// A term of frequency `freq` and document frequency `df`, in a
    /// document of length `len`.
    fn term(&self, freq: u32, df: usize, len: u32) -> f64 {
        let f = f64::from(freq);
        match self.scorer {
            Scorer::Bm25Std | Scorer::Bm25StdTanh => {
                bm25std(idf_bm25(self.n, df), f, len, self.avgdl)
            }
            Scorer::Bm25 => idf_tfidf(self.n, df) * bm25_legacy(f, self.avgdl),
            Scorer::Tfidf | Scorer::TfidfDocNorm => f * idf_tfidf(self.n, df),
            Scorer::DisMax => f,
            Scorer::DocScore | Scorer::Hamming => 0.0,
        }
    }

    /// What a result with no term scores: `*` (`freq` 1), or `NOT`
    /// (`freq` 0).
    fn virtual_value(&self, freq: u32, len: u32) -> f64 {
        if freq == 0 {
            return 0.0;
        }
        let f = f64::from(freq);
        match self.scorer {
            Scorer::Bm25Std | Scorer::Bm25StdTanh => bm25std(1.0, f, len, self.avgdl),
            Scorer::Bm25 => bm25_legacy(f, self.avgdl),
            Scorer::Tfidf | Scorer::TfidfDocNorm | Scorer::DisMax => f,
            Scorer::DocScore | Scorer::Hamming => 0.0,
        }
    }

    /// A NUMERIC (or vector range) clause that matched.
    fn numeric_value(&self) -> f64 {
        match self.scorer {
            Scorer::Bm25 => bm25_legacy(1.0, self.avgdl),
            Scorer::Tfidf | Scorer::TfidfDocNorm | Scorer::DisMax => 1.0,
            _ => 0.0,
        }
    }

    fn word(&self, def: &IndexDef, term: &str, stat: &Stat, len: u32) -> Res {
        let df = match self.term_df.get(term) {
            Some(df) => *df,
            None => def.doc_freq.read().get(term).copied().unwrap_or(0),
        };
        Res::leaf(
            Kind::Term,
            self.term(stat.freq, df, len),
            stat.offsets.clone(),
        )
    }

    fn aggregate(&self, kind: Kind, children: Vec<Res>) -> Option<Res> {
        if children.is_empty() {
            return None;
        }
        let value = if self.scorer == Scorer::DisMax && kind == Kind::Union {
            children.iter().map(|c| c.value).fold(0.0, f64::max)
        } else {
            children.iter().map(|c| c.value).sum()
        };
        let mut offsets: Vec<u32> = children
            .iter()
            .flat_map(|c| c.offsets.iter().copied())
            .collect();
        offsets.sort_unstable();
        Some(Res {
            kind,
            value,
            offsets,
            children,
        })
    }

    fn node(
        &self,
        def: &IndexDef,
        filter: &Filter,
        doc: &DocTerms,
        fields: &dyn Doc,
    ) -> Option<Res> {
        match filter {
            Filter::MatchNone => None,
            Filter::MatchAll => Some(Res::leaf(
                Kind::Virtual,
                self.virtual_value(1, doc.len),
                Vec::new(),
            )),
            Filter::Text {
                fields: scope,
                term,
            } => {
                let mask = fields_mask(def, scope);
                let present = |t: &str| doc.terms.get(t).filter(|s| s.fields & mask != 0);
                match term {
                    TextMatch::Exact(word) | TextMatch::Expanded(word) => {
                        present(word).map(|stat| self.word(def, word, stat, doc.len))
                    }
                    TextMatch::Stem(stem) => {
                        let term = format!("{STEM_PREFIX}{}", stem.stem);
                        present(&term).map(|stat| self.word(def, &term, stat, doc.len))
                    }
                    other => {
                        // RediSearch's union of the words a pattern expands
                        // to stops at the first that matches.
                        // Which one is not always the first in order, there:
                        // that is what is taken here.
                        let (word, stat) = doc
                            .terms
                            .iter()
                            .filter(|(t, s)| {
                                s.fields & mask != 0 && !is_stem(t) && other.matches(t)
                            })
                            .min_by_key(|(t, _)| **t)?;
                        let child = self.word(def, word, stat, doc.len);
                        self.aggregate(Kind::Union, vec![child])
                    }
                }
            }
            Filter::Phrase {
                fields: scope,
                terms,
            } => {
                let mask = fields_mask(def, scope);
                let mut children = Vec::new();
                for term in terms.iter().filter(|t| !def.is_stopword(t)) {
                    let stat = doc
                        .terms
                        .get(term.as_str())
                        .filter(|s| s.fields & mask != 0)?;
                    children.push(self.word(def, term, stat, doc.len));
                }
                self.aggregate(Kind::Intersection, children)
            }
            Filter::Tag {
                field,
                values,
                prefixes,
                separator,
                case_sensitive,
            } => {
                let text = fields
                    .field_bytes(field)
                    .and_then(|b| std::str::from_utf8(b).ok())?;
                let held: Vec<String> = super::filter::split_tags(text, *separator)
                    .map(|t| {
                        if *case_sensitive {
                            t.to_string()
                        } else {
                            t.to_lowercase()
                        }
                    })
                    .collect();
                let tag = |value: &str| {
                    let df = match self.tag_df.get(&(field.clone(), value.to_string())) {
                        Some(df) => *df,
                        None => def.postings.read().tag_docs(field, value),
                    };
                    Res::leaf(Kind::Term, self.term(1, df, doc.len), Vec::new())
                };
                let mut children: Vec<Res> = values
                    .iter()
                    .filter(|v| held.iter().any(|h| h == *v))
                    .map(|v| tag(v))
                    .collect();
                for prefix in prefixes {
                    if let Some(first) =
                        held.iter().filter(|h| h.starts_with(prefix.as_str())).min()
                    {
                        children.push(tag(first));
                    }
                }
                match children.len() {
                    0 => None,
                    1 if prefixes.is_empty() => children.pop(),
                    _ => self.aggregate(Kind::Union, children),
                }
            }
            Filter::Numeric { .. } | Filter::NumericNot { .. } | Filter::VectorRange { .. } => {
                super::filter::eval(filter, fields)
                    .then(|| Res::leaf(Kind::Numeric, self.numeric_value(), Vec::new()))
            }
            Filter::Not(inner) => match self.node(def, inner, doc, fields) {
                Some(_) => None,
                None if super::filter::eval(inner, fields) => None,
                None => Some(Res::leaf(Kind::Virtual, 0.0, Vec::new())),
            },
            Filter::Optional(inner) => Some(
                self.node(def, inner, doc, fields)
                    .unwrap_or_else(|| Res::leaf(Kind::Virtual, 0.0, Vec::new())),
            ),
            Filter::And(..) | Filter::Or(..) => {
                let and = matches!(filter, Filter::And(..));
                let mut parts = Vec::new();
                operands(filter, &mut parts);
                let mut children = Vec::with_capacity(parts.len());
                for part in parts {
                    match self.node(def, part, doc, fields) {
                        Some(res) => children.push(res),
                        None if and => return None,
                        None => {}
                    }
                }
                self.aggregate(if and { Kind::Intersection } else { Kind::Union }, children)
            }
        }
    }
}

/// The words, stems and TAG values a query names, for their document
/// frequencies. True if it has a pattern (a prefix, a fuzzy word, …), which
/// names no word in advance.
fn named_terms(filter: &Filter, words: &mut Vec<String>, tags: &mut Vec<(String, String)>) -> bool {
    match filter {
        Filter::Text { term, .. } => match term {
            TextMatch::Exact(w) | TextMatch::Expanded(w) => {
                words.push(w.clone());
                false
            }
            TextMatch::Stem(s) => {
                words.push(format!("{STEM_PREFIX}{}", s.stem));
                false
            }
            _ => true,
        },
        Filter::Phrase { terms, .. } => {
            words.extend(terms.iter().cloned());
            false
        }
        Filter::Tag { field, values, .. } => {
            tags.extend(values.iter().map(|v| (field.clone(), v.clone())));
            false
        }
        Filter::And(l, r) | Filter::Or(l, r) => {
            let left = named_terms(l, words, tags);
            named_terms(r, words, tags) || left
        }
        Filter::Not(inner) | Filter::Optional(inner) => named_terms(inner, words, tags),
        _ => false,
    }
}

/// RediSearch's "slop" of a result (`IndexResult_MinOffsetDelta`): the root
/// of the square of the sum of the distances between consecutive children's
/// nearest positions, or the number of children less one when none can be
/// measured. 1 for a result that is not a union or an intersection.
fn slop(res: &Res) -> u32 {
    const EOF: u32 = u32::MAX;
    if !matches!(res.kind, Kind::Union | Kind::Intersection) || res.children.len() <= 1 {
        return 1;
    }
    let children = &res.children;
    let num = children.len();
    let mut dist: i32 = 0;
    let mut i = 0;
    while i < num {
        while i < num && !children[i].has_offsets() {
            i += 1;
        }
        if i == num {
            break;
        }
        let v1 = &children[i].offsets;
        i += 1;
        while i < num && !children[i].has_offsets() {
            i += 1;
        }
        if i == num {
            break;
        }
        let v2 = &children[i].offsets;
        let (mut a, mut b) = (0usize, 0usize);
        let at = |v: &Vec<u32>, k: usize| v.get(k).copied().unwrap_or(EOF);
        let (mut p1, mut p2) = (at(v1, 0), at(v2, 0));
        // As in C: an unsigned distance put in an int.
        let delta = |x: u32, y: u32| x.abs_diff(y) as i32;
        let mut cd = delta(p2, p1);
        while cd > 1 && p1 != EOF && p2 != EOF {
            cd = cd.min(delta(p2, p1));
            if p2 > p1 {
                a += 1;
                p1 = at(v1, a);
            } else {
                b += 1;
                p2 = at(v2, b);
            }
        }
        dist = dist.wrapping_add(cd.wrapping_mul(cd));
    }
    if dist > 0 {
        (f64::from(dist).sqrt() as u32).max(1)
    } else if dist == 0 {
        (num - 1) as u32
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_as_redisearch_has_it() {
        // Four documents, a word in two of them: ln(1 + 2.5 / 2.5).
        assert!((idf_bm25(4, 2) - 2f64.ln()).abs() < 1e-12);
        // N + 1 over df, base 2, rounded down: 1 + 6/2 = 4 → 2.
        assert_eq!(idf_tfidf(5, 2), 2.0);
        assert_eq!(idf_tfidf(2, 2), 1.0);
    }

    #[test]
    fn scorers_by_name() {
        assert_eq!(Scorer::parse("bm25std"), Some(Scorer::Bm25Std));
        assert_eq!(Scorer::parse("TFIDF.DOCNORM"), Some(Scorer::TfidfDocNorm));
        assert_eq!(Scorer::parse("nope"), None);
    }
}
