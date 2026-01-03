## Spotify Lyric Search

I built a lyric-based song identification system that predicts the song title and artist
from a short snippet of lyrics.

### Approach
- Text preprocessing (cleaning + stopword removal)
- TF-IDF vectorization using character n-grams
- Cosine similarity for lyric matching

### Why similarity instead of deep learning?
The dataset is large but static. A similarity-based approach is faster,
interpretable, and performs well for partial lyric queries.

### Example
Input: "hello darkness my old friend"  
Output: Simon & Garfunkel – The Sound of Silence
