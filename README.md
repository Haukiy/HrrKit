## IEEE source formatting 

The `Scource in IDEEE` column accepts either a fully formatted IEEE reference or
an identifier that the script can expand on your behalf:

- **DOIs** – accepted forms: `doi:10.1109/5.771073`, `https://doi.org/10.1109/5.771073`, or a raw `10.xxxx/yyy` string.
- **ISBNs** – accepted forms: `isbn:9780131101630`, `ISBN 0-201-53082-1`, or a plain 10/13 digit value with optional hyphens.

When one of these patterns is detected the script resolves the identifier and
replaces the `Scource in IDEEE` value with an IEEE-formatted reference. This
updated value is stored in `database.csv`, the per-topic source registry, and the
plot captions, so the user only has to provide the DOI/ISBN once.

