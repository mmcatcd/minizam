# minizam
A very small audio fingerprinting and search library inspired by Shazam

## Example Usage

```python
from minizam.engine import Engine
e = Engine()
e.seed_database([
    {"path": "./audio/do_it_again.wav", "name": "Do it again"},
    {"path": "./audio/signed_sealed_delivered.wav", "name": "Signed, Sealed & Delivered"},
])
matches, max_matches, max_file_name = e.match("./audio/do_it_again_sample.wav")
```
