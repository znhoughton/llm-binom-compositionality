# Sentence Pool: Generation and Exclusion Criteria

## Generation

For each attested binomial (word1, word2) we generated up to 500 sentences
containing the AB ordering ("word1 and word2") via the Claude API.  The BA
ordering ("word2 and word1") is **never generated independently**: every BA
sentence is derived by swapping the phrase in-place in its corresponding AB
sentence.  This ensures the only difference between paired AB and BA sentences
is word order — sentence context is held constant.

Sentences are lowercased and space-prepended before tokenisation so that every
word in the phrase receives a space-prefix BPE token (the mid-sentence form),
regardless of whether the phrase falls at the start of the sentence.

## Exclusion criteria

Sentences were excluded (both the AB sentence and its paired BA sentence) if
any of the following conditions held.  All checks are applied after lowercasing
and space-prepending (i.e. in the form passed to the tokeniser).

| Criterion | Reason |
|---|---|
| Sentence shorter than 5 words | Not a real sentence (e.g. a markdown heading `# Fruits and Vegetables`) |
| Sentence starts with `#` | LLM-generated section header, not a usable context |
| Phrase absent from sentence | Generation artefact; span finder would fail |
| BA phrase present in AB sentence | Both orderings in the same sentence; context confound |
| Phrase occurs more than once | Span finder captures only the first occurrence; second occurrence adds noise |
| Duplicate sentence within a phrase | Redundant context |
| Context mismatch after swap | Swap regex matched inside a compound word or quoted string (e.g. *paint**brush** and trees*, *"pepper and salt"*), producing a garbled BA sentence |
| Phrase token multiset differs between AB and BA | Same root cause as context mismatch: phrase boundary is not a space, so one word tokenises without a space prefix and the token IDs differ |

## Verification

After all exclusions, the following invariants were confirmed across all
296,866 AB/BA sentence pairs using the OPT tokeniser:

- **Context tokens identical**: the token IDs outside the phrase span are
  bit-for-bit identical between AB and BA (0 mismatches).
- **Phrase token multiset identical**: the unordered set of phrase token IDs
  is the same for AB and BA — only the order differs (0 mismatches).
- **Span always found**: the phrase span is successfully located in every
  sentence (0 failures).

Final sentence counts per binomial: 480 × 500, 96 × 499, 16 × 498, 2 × 497.
