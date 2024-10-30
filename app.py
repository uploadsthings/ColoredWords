from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re
import nltk
from nltk.corpus import cmudict
import pronouncing
from typing import Dict, Set, Tuple, List
import colorsys
from collections import OrderedDict, defaultdict
import random
from nltk.corpus import wordnet as dictionary
import os
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

try:
    nltk.data.find('corpora/cmudict')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    d = cmudict.dict()
except LookupError as e:
    logger.error(f"NLTK data not found: {e}")
    logger.info("Attempting to download NLTK data...")
    try:
        nltk.download('cmudict')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        d = cmudict.dict()
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        raise

class SuggestionEngine:
    def __init__(self):
        self.cache = {
            'rhymes': {},
            'synonyms': {}
        }

    def clean_word(self, word: str) -> str:
        return re.sub(r'[^\w\s]', '', word.lower())

    def get_rhyming_words(self, word: str, max_suggestions: int = 10) -> List[str]:
        word = self.clean_word(word)
        
        if word in self.cache['rhymes']:
            return self.cache['rhymes'][word]
        
        rhymes = pronouncing.rhymes(word)
        
        filtered_rhymes = [
            rhyme for rhyme in rhymes 
            if rhyme != word and 
            not word.endswith(rhyme) and 
            not rhyme.endswith(word) and
            len(rhyme) >= 2
        ]
        
        # Add error handling for WordNet lookups
        scored_rhymes = []
        for rhyme in filtered_rhymes:
            try:
                frequency_score = len(dictionary.synsets(rhyme))
                scored_rhymes.append((rhyme, frequency_score))
            except:
                # If WordNet lookup fails, use a default score of 1
                scored_rhymes.append((rhyme, 1))
        
        scored_rhymes.sort(key=lambda x: (-x[1], len(x[0])))
        result = [word for word, _ in scored_rhymes[:max_suggestions]]
        
        if len(result) > 5:
            static_suggestions = result[:5]
            random_suggestions = random.sample(result[5:], min(5, len(result[5:])))
            result = static_suggestions + random_suggestions
        
        self.cache['rhymes'][word] = result
        return result
    
    def get_synonyms(self, word: str, max_suggestions: int = 10) -> List[str]:
        word = self.clean_word(word)
        
        if word in self.cache['synonyms']:
            return self.cache['synonyms'][word]
        
        synonyms: Dict[str, float] = {}  # word -> score
        word_synsets = dictionary.synsets(word)

        if not word_synsets:
            return []  # Early return if there are no synsets for the word

        # Use the most common sense as the base context
        primary_sense = word_synsets[0]

        for syn in word_synsets:
            base_score = 1.0

            try:
                similarity = syn.wup_similarity(primary_sense)
                if similarity:
                    base_score = similarity
            except:
                pass

            pos_multiplier = 1.2 if syn.pos() == primary_sense.pos() else 0.8

            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word and ' ' not in synonym:
                    # Score based on:
                    # 1. similarity (context relevance)
                    # 2. part of speech match
                    # 3. frequency
                    frequency_score = len(dictionary.synsets(synonym))
                    total_score = base_score * pos_multiplier * (1 + frequency_score / 10)

                    if synonym in synonyms:
                        synonyms[synonym] = max(synonyms[synonym], total_score)
                    else:
                        synonyms[synonym] = total_score

        scored_synonyms = sorted(synonyms.items(), key=lambda x: -x[1])
        result = [word for word, _ in scored_synonyms[:max_suggestions]]

        if len(result) > 5:
            static_suggestions = result[:5]
            random_suggestions = random.sample(result[5:], min(5, len(result[5:])))
            result = static_suggestions # + random_suggestions

        # Cache the result
        self.cache['synonyms'][word] = result
        return result

    def refresh_suggestions(self, word: str) -> Tuple[List[str], List[str]]:
        """Get fresh rhyming words and synonyms for the given word."""
        rhymes = self.get_rhyming_words(word)
        synonyms = self.get_synonyms(word)
        return rhymes, synonyms
    
suggestion_engine = SuggestionEngine()

class RhymeAnalyzer:
    def __init__(self):
        self.rhyme_groups: Dict[str, List[str]] = {}
        self.color_schemes: Dict[str, str] = {}
        self.word_occurrences: Dict[str, int] = {}  
        self.next_color_index = 0
        
        self.base_colors = [
            (250, 70, 64),  # red
            (129, 236, 236),  # cyan
            (248, 171, 231),  # salmon
            (85, 239, 196),   # green
            (253, 121, 168),  # pink
            (0, 206, 201),    # teal
            (184, 146, 255),  # rose
            (46, 134, 222),   # sky blue
            (225, 112, 85),   # coral
            (123, 237, 159),  # mint green
            (255, 107, 129),  # soft salmon
            (116, 185, 255)   # light blue
        ]

    def is_valid_rhyme(self, word1: str, word2: str, rhyme_phonemes: List[str]) -> bool:
        """
        Check if two words form a valid rhyme pair.
        """
        if word1 == word2:
            return False
            
        common_suffixes = {'ing', 'ed', 'ly'}  
        for suffix in common_suffixes:
            if word1.endswith(suffix) and word2.endswith(suffix):
                if word1[:-len(suffix)] == word2[:-len(suffix)]:
                    return False
        
        word1_phones = pronouncing.phones_for_word(word1)
        word2_phones = pronouncing.phones_for_word(word2)
        
        if not word1_phones or not word2_phones:
            return False
            
        return True

    def get_rhyme_phonemes(self, word: str) -> Tuple[List[str], List[str]]:
        """Returns both full phonemes and rhyming phonemes for a word."""
        word = word.lower()
        if word in d:
            full_phonemes = d[word][0]
            vowel_positions = [i for i, phoneme in enumerate(full_phonemes) 
                             if any(char.isdigit() for char in phoneme)]
            
            if vowel_positions:
                last_vowel_pos = vowel_positions[-1]
                rhyme_phonemes = full_phonemes[last_vowel_pos:]
                return full_phonemes, rhyme_phonemes
        return [], []

    def generate_color(self, base_color: Tuple[int, int, int], variation: int = 0) -> str:
        """Generate a color variation based on a base color."""
        r, g, b = base_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        s = min(1.0, s + (variation * 0.1))
        v = min(1.0, v + (variation * 0.1))
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def analyze_poem(self, poem: str) -> Dict[str, dict]:
        """Analyze the poem and create rhyme groups with color coding."""
        lines = poem.strip().split('\n')
        word_metadata = OrderedDict()  
        temp_rhyme_groups: Dict[str, List[str]] = {}
        
        word_positions = {} 
        current_position = 0
        
        for line in lines:
            words = line.split()
            for word in words:
                clean_word = re.sub(r'[^\w\s]', '', word.lower())
                if not clean_word:
                    continue
                
                if clean_word not in word_positions:
                    word_positions[clean_word] = []
                word_positions[clean_word].append(current_position)
                current_position += 1
                
                full_phonemes, rhyme_phonemes = self.get_rhyme_phonemes(clean_word)
                if rhyme_phonemes:
                    rhyme_key = '-'.join(rhyme_phonemes)
                    if rhyme_key not in temp_rhyme_groups:
                        temp_rhyme_groups[rhyme_key] = []
                    if clean_word not in temp_rhyme_groups[rhyme_key]:
                        temp_rhyme_groups[rhyme_key].append(clean_word)
        
        valid_rhyme_groups: Dict[str, List[str]] = {}
        
        for rhyme_key, words in temp_rhyme_groups.items():
            if len(set(words)) < 2:
                continue
                
            valid_words = []
            rhyme_phonemes = rhyme_key.split('-')
            
            for word1 in words:
                has_valid_rhyme = False
                for word2 in words:
                    if word1 != word2 and self.is_valid_rhyme(word1, word2, rhyme_phonemes):
                        has_valid_rhyme = True
                        break
                if has_valid_rhyme and word1 not in valid_words:
                    valid_words.append(word1)
            
            if len(valid_words) >= 2:
                valid_rhyme_groups[rhyme_key] = valid_words
                
                if rhyme_key not in self.color_schemes:
                    base_color = self.base_colors[self.next_color_index % len(self.base_colors)]
                    self.color_schemes[rhyme_key] = self.generate_color(base_color)
                    self.next_color_index += 1
        
        for rhyme_key, words in valid_rhyme_groups.items():
            for word in words:
                first_position = word_positions[word][0]
                word_phonemes = pronouncing.phones_for_word(word)
                if word_phonemes:
                    rhyme_phonemes = rhyme_key.split('-')
                    rhyme_chars = len(''.join(rhyme_phonemes))
                    non_rhyme_chars = max(0, len(word) - rhyme_chars)
                    
                    word_metadata[first_position] = {
                        'word': word,
                        'rhyme_key': rhyme_key,
                        'color': self.color_schemes[rhyme_key],
                        'non_rhyme_length': non_rhyme_chars,
                        'full_word': word
                    }
        
        return OrderedDict(sorted(word_metadata.items()))

def generate_poem_html(poem: str, analyzer: RhymeAnalyzer) -> str:
    """Generate HTML for the poem with rhyme highlighting."""
    word_metadata = analyzer.analyze_poem(poem)
    lines = poem.strip().split('\n')
    poem_html = []
    current_position = 0
    
    for line in lines:
        line_html = []
        words = line.split()
        
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            
            if current_position in word_metadata and clean_word == word_metadata[current_position]['word']:
                metadata = word_metadata[current_position]
                non_rhyme_length = metadata['non_rhyme_length']
                color = metadata['color']
                rhyme_key = metadata['rhyme_key']
                
                # Split word into non-rhyming and rhyming parts
                non_rhyme_part = word[:non_rhyme_length]
                rhyme_part = word[non_rhyme_length:]
                
                # Add data-word attribute to the entire word span
                word_html = f'<span data-word="{clean_word}">{non_rhyme_part}'
                word_html += ''.join([
                    f'<span class="rhyme-char rhyme-group-{rhyme_key}" '
                    f'style="color: {color};">{char}</span>'
                    for char in rhyme_part
                ])
                word_html += '</span>'
                
                line_html.append(word_html)
            else:
                line_html.append(word)
            
            current_position += 1
        
        poem_html.append(' '.join(line_html))
    
    return '<br>'.join(poem_html)

@app.route('/get_suggestions', methods=['POST'])
@limiter.limit("30 per minute")
def get_suggestions():
    """Handle requests for word suggestions."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
            
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({
                'rhymes': [],
                'synonyms': []
            })
        
        rhymes = suggestion_engine.get_rhyming_words(word)
        synonyms = suggestion_engine.get_synonyms(word)
        
        return jsonify({
            'rhymes': rhymes,
            'synonyms': synonyms
        })
    except Exception as e:
        logger.error(f"Error in get_suggestions: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return "Error loading application", 500

@app.route('/update_poem', methods=['POST'])
@limiter.limit("60 per minute")
def update_poem():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
            
        poem = data.get("poem", "")
        analyzer = RhymeAnalyzer()
        poem_html = generate_poem_html(poem, analyzer)
        return jsonify({"poem_html": poem_html})
    except Exception as e:
        logger.error(f"Error in update_poem: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
