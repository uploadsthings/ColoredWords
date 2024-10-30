import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK data."""
    required_corpora = [
        'cmudict',
        'wordnet',
        'omw-1.4'
    ]

    for corpus in required_corpora:
        try:
            nltk.download(corpus)
            print(f"Successfully downloaded {corpus}")
        except Exception as e:
            print(f"Error downloading {corpus}: {str(e)}")

if __name__ == "__main__":
    download_nltk_data()
