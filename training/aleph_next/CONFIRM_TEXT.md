# Fresh text confirmation at 14M

The frozen panel contains 384 previously unscored 33-byte windows from twelve authored books. The first byte supplies context for the first prediction. Each document contributes 1,024 equally weighted next-byte losses. The public commitment to the private manifest is `82761739ea4f078bd1631dba2851be6551109692fbfe5166295429094fe45c51`.

| Stratum | Four independent author-documents |
| --- | --- |
| Fiction | Jane Austen, *Pride and Prejudice* (1342)<br>Mary Shelley, *Frankenstein* (84)<br>Arthur Conan Doyle, *A Study in Scarlet* (244)<br>Robert Louis Stevenson, *The Strange Case of Dr. Jekyll and Mr. Hyde* (43) |
| Observation and science | Charles Darwin, *On the Origin of Species* (1228)<br>Henry David Thoreau, the *Walden* edition (205)<br>John Muir, *The Mountains of California* (10012)<br>Michael Faraday, *The Chemical History of a Candle* (14474) |
| Memoir and social argument | Frederick Douglass, *Narrative of the Life* (23)<br>Benjamin Franklin, *The Autobiography* (148)<br>Mary Wollstonecraft, *A Vindication of the Rights of Woman* (3420)<br>W. E. B. Du Bois, *The Souls of Black Folk* (408) |

The numbers identify Project Gutenberg books. Each source record pins the GITenberg repository, commit, plaintext path and SHA-256, and the matching Gutenberg RDF catalogue path and SHA-256. Every RDF record identifies the text as public domain in the USA. Several later illustrated editions have separately licensed covers. The panel uses the plaintext. [Edition and catalogue pins](confirm_text_sources.json)

## Selection and prior overlap

Decode the declared character encoding strictly, strip a UTF-8 BOM if present, and normalize newlines. For the older Wollstonecraft edition, the end of the old legal preamble marks the beginning of the book. Remove the Gutenberg wrappers, normalize Unicode to NFC, replace each run of whitespace with one ASCII space and encode as UTF-8. Use the central 60% of each resulting body to limit front and back matter. Place nonoverlapping 33-byte candidates on a fixed grid, require valid UTF-8, at least 15 letters and at least three spaces. Divide the central span into 32 equal byte bands.

The architecture worker confirmed the only natural-language training and development inputs were the pinned native 512 train and 128 development windows. Reject a candidate sharing any contiguous 16 bytes with those windows, either in the original bytes or after NFKC case folding, removing punctuation and joining alphanumeric words with spaces. The old records lack source-document IDs, which limits the overlap check to the bytes actually observed by training and development. An independent implementation found the longest shared substring was 15 bytes and occurred in two of the 384 selected windows.

Within each band, rank the surviving offsets by HMAC-SHA256 of the document ID and offset under one privately drawn 256-bit key. Select the lowest rank whose original and folded 16-byte spans also avoid all earlier selected windows. The key and windows stay with the evaluator. Of 96,586 grid windows, 95,863 met the prose rule and 294 overlapped native data. This left 95,569 candidates, with 62 to 529 in each band. Four candidates encountered an earlier selected overlap. The offline replay reproduced the same manifest and exact windows.

## Scoring

Use the native causal byte readout on the full vocabulary. Score all 32 next-byte positions per window and equally weight the twelve document means. Report nats per byte. A native-training-only add-one unigram and an interpolated trigram provide reference losses. The latter shrinks each conditional distribution toward the shorter context with eight pseudo-observations and resets context at every source window. Only the first byte of the current window precedes its first target.

The standalone absolute-loss diagnostic computes paired B-minus-A differences within document and matched seed. Average within document over the fixed seeds, then resample four whole documents with replacement within each of the three fixed strata. Use 10,000 draws, bootstrap seed 37019 and an interpolated percentile 95% interval. The tournament's joint gate separately computes relative gain with the same document blocks and its registered 20,000 paired draws. Also retain all original per-document, per-stratum and per-seed differences. The twelve authors are a purposive panel of historical English-language prose. Its interval describes document sensitivity conditional on those strata and seeds. Technical web text, newer linguistic registers, training-seed uncertainty and full-scale behavior require independent evidence. [Joint decision protocol](CONFIRM14_PROTOCOL.md)

```sh
python training/aleph_next/confirm_text.py fetch --cache ../aleph-confirm14/source-cache
python training/aleph_next/confirm_text.py freeze --cache ../aleph-confirm14/source-cache --pack runs/placement/mixed --panel ../aleph-confirm14/new-text-panel
python training/aleph_next/confirm_text.py audit --cache ../aleph-confirm14/source-cache --pack runs/placement/mixed --panel ../aleph-confirm14/text
```

The first two commands build an independent draw. The final command audits the original custodied draw using its saved key. The program refuses to replace an existing panel. The checkpoint evaluator releases scores only after the architecture and competence gates permit opening the panel. [Panel builder and paired scorer](confirm_text.py)
