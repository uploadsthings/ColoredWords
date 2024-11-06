import os
from flask import Flask, render_template_string, request, jsonify
import re
import nltk
from nltk.corpus import cmudict
import pronouncing
from typing import Dict, Set, Tuple, List, Optional
import colorsys
from collections import OrderedDict, defaultdict
import random
from nltk.corpus import wordnet as dictionary
from nltk.corpus.reader.wordnet import Synset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('cmudict')
nltk.download('wordnet')
nltk.download('omw-1.4')
d = cmudict.dict()

app = Flask(__name__)

class SuggestionEngine:
    def __init__(self):
        self.cache = {
            'rhymes': {},
            'synonyms': {}
        }

    def clean_word(self, word: str) -> str:
        return re.sub(r'[^\w\s]', '', word.lower())
        
    def is_valid_word(self, word: str) -> bool:
        print(f"\nValidating word: {word}")
        
        if not word.isalpha() or ' ' in word:
            return False
            
        if word not in d:
            return False
            
        synsets = dictionary.synsets(word)
        if not synsets:
            return False
            
        if len(word) > 15:
            return False
            
        total_count = sum(
            lemma.count()
            for synset in synsets 
            for lemma in synset.lemmas()
            if lemma.name().lower() == word
        )
        
        return True

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
        
        scored_rhymes = []
        for rhyme in filtered_rhymes:
            try:
                frequency_score = len(dictionary.synsets(rhyme))
                scored_rhymes.append((rhyme, frequency_score))
            except:
                scored_rhymes.append((rhyme, 1))
        
        scored_rhymes.sort(key=lambda x: (-x[1], len(x[0])))
        result = [word for word, _ in scored_rhymes[:max_suggestions]]
        
        if len(result) > 5:
            static_suggestions = result[:5]
            random_suggestions = random.sample(result[5:], min(5, len(result[5:])))
            result = static_suggestions + random_suggestions
        
        self.cache['rhymes'][word] = result
        return result
    
    def get_true_synonyms(self, word: str) -> Set[str]:
        word = self.clean_word(word)
        synonyms = set()
        
        word_synsets = dictionary.synsets(word)
        
        for synset in word_synsets:
            similarity = max(
                w1.path_similarity(w2) or 0
                for w1 in word_synsets
                for w2 in [synset]
            )
            
            if similarity > 0.1:  
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word and ' ' not in synonym:
                        synonyms.add(synonym)
                
        return synonyms

    def get_likely_sense(self, word: str, context_words: List[str]) -> Optional[Synset]:
        synsets = dictionary.synsets(word)
        if not synsets:
            return None
            
        if len(synsets) == 1:
            return synsets[0]
        
        max_similarity = 0
        best_synset = synsets[0]
        
        for synset in synsets:
            context = (synset.definition() + ' ' + 
                      ' '.join(synset.examples())).split()
            
            similarity = len(set(context) & set(context_words))
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_synset = synset
                
        return best_synset

    def get_rhyming_synonyms(self, word: str, whole_poem: str, max_suggestions: int = 20) -> List[str]:
        logs = [f"\n=== Finding rhyming synonyms for '{word}' ==="]
        word = self.clean_word(word)
        
        if word in self.cache['synonyms']:
            logs.append("📦 Using cached results")
            print('\n'.join(logs))
            return self.cache['synonyms'][word]

        context_words = [self.clean_word(w) for w in whole_poem.split()]
        
        poem_rhyme_patterns = set()
        for poem_word in context_words:
            if poem_word != word:
                phones = pronouncing.phones_for_word(poem_word)
                if phones:
                    poem_rhyme_patterns.add(self._get_rhyming_phones(phones[0]))

        likely_sense = self.get_likely_sense(word, context_words)
        
        synonyms = set()
        if likely_sense:
            for lemma in likely_sense.lemmas():
                synonym = lemma.name().lower()
                if synonym != word and ' ' not in synonym:
                    synonyms.add(synonym)
                    
            for synset in dictionary.synsets(word):
                if synset != likely_sense:
                    similarity = likely_sense.path_similarity(synset)
                    if similarity and similarity > 0.2:
                        for lemma in synset.lemmas():
                            synonym = lemma.name().lower()
                            if synonym != word and ' ' not in synonym:
                                synonyms.add(synonym)
        
        logs.append(f"\nFound {len(synonyms)} initial synonyms: {sorted(synonyms)}")

        # Score valid synonyms that rhyme
        scored_synonyms = []
        validation_logs = []
        discarded_synonyms = {
            'invalid': [],
            'no_pronunciation': [],
            'no_rhyme': []
        }
        
        for synonym in synonyms:
            validation_logs.append(f"\nValidating word: {synonym}")
            if not self.is_valid_word(synonym):
                discarded_synonyms['invalid'].append(synonym)
                continue
                    
            syn_phones = pronouncing.phones_for_word(synonym)
            if not syn_phones:
                discarded_synonyms['no_pronunciation'].append(synonym)
                continue
                    
            syn_rhyme_pattern = self._get_rhyming_phones(syn_phones[0])
            if syn_rhyme_pattern not in poem_rhyme_patterns:
                discarded_synonyms['no_rhyme'].append(synonym)
                continue

            # Calculate score based on usage frequency and length similarity
            score = 0.0
            freq_count = sum(
                lemma.count() 
                for synset in dictionary.synsets(synonym)
                for lemma in synset.lemmas()
                if lemma.name().lower() == synonym
            )
            
            if freq_count > 0:
                score += min(freq_count / 5.0, 2.0)
                
            length_diff = abs(len(synonym) - len(word))
            if length_diff <= 2:
                score += 0.5
            elif length_diff <= 4:
                score += 0.3

            scored_synonyms.append((synonym, score))

        # Add validation results to log
        logs.extend(validation_logs)

        logs.extend([
            "\nDiscarded words:",
            f"- Failed validation: {discarded_synonyms['invalid']}",
            f"- No pronunciation: {discarded_synonyms['no_pronunciation']}",
            f"- Doesn't rhyme with poem: {discarded_synonyms['no_rhyme']}"
        ])

        scored_synonyms.sort(key=lambda x: (-x[1], len(x[0])))
        
        logs.append("\nScored synonyms:")
        for word, score in scored_synonyms:
            logs.append(f"- {word}: {score:.2f}")
        
        # Get final selections - take all valid rhyming synonyms
        best_synonyms = [word for word, _ in scored_synonyms[:max_suggestions]]
        logs.append(f"\nFinal suggestions: {best_synonyms}")

        # Print all logs at once at the end
        print('\n'.join(logs))

        self.cache['synonyms'][word] = best_synonyms
        return best_synonyms

    def _get_rhyming_phones(self, phones: str) -> str:
        phones = phones.split()
        for i, phone in enumerate(phones):
            if any(char.isdigit() for char in phone):  
                last_vowel_pos = i
        return ' '.join(phones[last_vowel_pos:])

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

    def get_syllable_stress_pattern(self, word: str) -> List[int]:
        if word not in d:
            return []
        phones = d[word][0]
        stresses = []
        for phone in phones:
            if any(char.isdigit() for char in phone):
                stress = int(next(char for char in phone if char.isdigit()))
                stresses.append(stress)
        return stresses

    def analyze_meter(self, line: str) -> str:
        words = line.split()
        total_feet = 0
        stress_pattern = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            stresses = self.get_syllable_stress_pattern(clean_word)
            stress_pattern.extend(stresses)
            total_feet += len(stresses)
        
        if len(stress_pattern) >= 2:
            if stress_pattern[::2] == [0] * (len(stress_pattern)//2):
                return f"Iambic ({total_feet//2} feet)"
            elif stress_pattern[::2] == [1] * (len(stress_pattern)//2):
                return f"Trochaic ({total_feet//2} feet)"
        
        return f"{total_feet} syllables"

    def is_valid_rhyme(self, word1: str, word2: str, rhyme_phonemes: List[str]) -> bool:
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

def get_syllable_stress_pattern(self, word: str) -> List[int]:
    """(0=unstressed, 1=primary, 2=secondary)"""
    if word not in d:
        return []
    phones = d[word][0]
    stresses = []
    for phone in phones:
        if any(char.isdigit() for char in phone):
            stress = int(next(char for char in phone if char.isdigit()))
            stresses.append(stress)
    return stresses

def generate_poem_html(poem: str, analyzer: RhymeAnalyzer) -> str:
    word_metadata = analyzer.analyze_poem(poem)
    lines = poem.strip().split('\n')
    poem_html = []
    current_position = 0

    for line in lines:
        line_html = []
        words = line.split()

        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word.lower())

            stress_markers = ''
            if clean_word:
                stresses = analyzer.get_syllable_stress_pattern(clean_word)
                if stresses:
                    stress_markers = '<div class="stress-markers">' + ''.join(
                        '<span class="stress-mark stressed"></span>' if stress > 0 else '<span class="stress-mark"></span>'
                        for stress in stresses
                    ) + '</div>'

            if current_position in word_metadata and clean_word == word_metadata[current_position]['word']:
                metadata = word_metadata[current_position]
                non_rhyme_length = metadata['non_rhyme_length']
                color = metadata['color']
                rhyme_key = metadata['rhyme_key']

                word_html = f'<span class="word-wrapper" data-word="{clean_word}">'
                word_html += f'<span class="word-text">{word[:non_rhyme_length]}'
                word_html += ''.join([
                    f'<span class="rhyme-char rhyme-group-{rhyme_key}" '
                    f'style="color: {color};">{char}</span>'
                    for char in word[non_rhyme_length:]
                ])
                word_html += '</span>'
                word_html += stress_markers
                word_html += '</span>'
            else:
                word_html = f'<span class="word-wrapper"><span class="word-text">{word}</span>{stress_markers}</span>'

            line_html.append(word_html)
            current_position += 1

        poem_html.append(' '.join(line_html))

    return '\n'.join(poem_html)

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    """Handle requests for word suggestions."""
    data = request.get_json()
    word = data.get('word', '').strip()
    poem_text = data.get('poem', '').strip()
    
    if not word:
        return jsonify({
            'rhymes': [],
            'synonyms': []
        })
    
    rhymes = suggestion_engine.get_rhyming_words(word)
    synonyms = suggestion_engine.get_rhyming_synonyms(word, poem_text)
    
    return jsonify({
        'rhymes': rhymes,
        'synonyms': synonyms
    })

@app.route('/')
def index():
    try:
        logger.info("Attempting to render index.html")
        # template exists
        if not os.path.exists(os.path.join(app.template_folder, 'index.html')):
            logger.error("index.html not found in templates folder")
            return "Error: Template not found", 404
            
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}", exc_info=True)
        return f"Error loading application: {str(e)}", 500

@app.route('/update_poem', methods=['POST'])
def update_poem():
    try:
        data = request.get_json()
        poem = data.get("poem", "")
        analyzer = RhymeAnalyzer()
        poem_html = generate_poem_html(poem, analyzer)
        return jsonify({"poem_html": poem_html})
    except Exception as e:
            logger.error(f"Error in update_poem: {e}")
            return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
