#!/usr/bin/env python3
"""
Audio Processing Script for Echoes Installation

This script processes audio files by:
1. Transcribing audio using Whisper
2. Generating a story from the transcription using GPT-5
3. Creating audio from the generated story using the original audio segments
4. Optionally generating a wordcloud from the transcription
"""

import argparse
import json
import random
import string
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import whisper_timestamped as whisper
import wordcloud
from dotenv import load_dotenv
from openai import OpenAI


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def find_start_end(sentence: str, all_words: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the start and end time that perfectly match a given sentence.
    
    Args:
        sentence: The sentence to find the start and end time for
        all_words: List of word dictionaries with timing information
        
    Returns:
        Tuple of (start_time, end_time) or (None, None) if not found
    """
    words_in_sentence = sentence.split(" ")
    
    for index, word in enumerate(all_words):
        found = True
        clean_first_word = words_in_sentence[0].strip().lower().translate(
            str.maketrans('', '', string.punctuation)
        )
        
        if word["text"] == clean_first_word:
            start, end = word["start"], word["end"]
            
            for i in range(1, len(words_in_sentence)):
                clean_word = words_in_sentence[i].strip().lower().translate(
                    str.maketrans('', '', string.punctuation)
                )
                
                if clean_word == all_words[index + i]["text"]:
                    end = all_words[index + i]["end"]
                elif (i < len(words_in_sentence) - 1 and 
                      clean_word == all_words[index + i]["text"] + all_words[index + i + 1]["text"]):
                    # Word is split into two parts
                    end = all_words[index + i + 1]["end"]
                    index += 1
                elif (i < len(words_in_sentence) - 2 and 
                      clean_word == all_words[index + i]["text"] + all_words[index + i + 1]["text"] + all_words[index + i + 2]["text"]):
                    # Word is split into three parts
                    end = all_words[index + i + 2]["end"]
                    index += 2
                else:
                    found = False
                    break
                    
            if found:
                return start, end
                
    return None, None


def find_longest_sequence(sentence: str, results: dict) -> Tuple[str, Tuple[str, str]]:
    """
    Find the longest sequence of words that are in the transcription text.
    
    Args:
        sentence: The sentence to find sequences in
        results: Whisper transcription results
        
    Returns:
        Tuple of (longest_sequence, (before_text, after_text))
    """
    words_in_sentence = sentence.split(" ")
    
    # Generate all possible sequences of words
    sequences = []
    for i in range(len(words_in_sentence)):
        for j in range(i + 1, len(words_in_sentence) + 1):
            sequences.append(" ".join(words_in_sentence[i:j]))
    
    # Find the longest sequence that is in the text
    longest_sequence = ""
    for sequence in sequences:
        test = (f" {sequence} " in results["text"] or 
                f" {sequence}. " in results["text"] or 
                f" {sequence}, " in results["text"])
        if test and len(sequence) > len(longest_sequence):
            longest_sequence = sequence
    
    if longest_sequence == "":
        return "", ("", "")
    
    # Return the longest sequence and whatever is before and after it
    sentence_split = sentence.split(longest_sequence)
    before = sentence_split[0]
    after = " ".join(sentence_split[1:])
    return longest_sequence, (before.strip(), after.strip())


def crossfade(audio: np.ndarray) -> np.ndarray:
    """
    Create a quick crossfade at the beginning and end of audio.
    
    Args:
        audio: Audio array to process
        
    Returns:
        Audio array with crossfade applied
    """
    fade_length = min(80000, int(len(audio) / 30))
    fade = np.linspace(0, 1, fade_length)
    audio[:fade_length] = audio[:fade_length] * fade
    audio[-fade_length:] = audio[-fade_length:] * fade[::-1]
    return audio


def cover_sentence(sentence: str, audio: np.ndarray, sr: int, all_words: List[dict], results: dict) -> List[float]:
    """
    Recursively cover the entire sentence with longest sequences.
    Build the audio by concatenating the audio of the longest sequences.
    
    Args:
        sentence: Sentence to process
        audio: Original audio array
        sr: Sample rate
        all_words: List of word dictionaries
        results: Whisper transcription results
        
    Returns:
        List of audio samples for the sentence
    """
    if sentence == "":
        return []
        
    longest_sequence, (before, after) = find_longest_sequence(sentence, results)
    if longest_sequence == "":
        return []
        
    start, end = find_start_end(longest_sequence, all_words)
    if start is None or end is None:
        return []
        
    print(f"  Processing: '{longest_sequence}' (time: {start:.2f}s - {end:.2f}s)")
    
    audio_building = []
    audio_building.extend(crossfade(audio[int(start * sr):int((end + 0.1 * (end - start)) * sr)]))
    audio_building.extend(np.zeros(int(0.1 * (end - start) * sr)))
    
    return (cover_sentence(before, audio, sr, all_words, results) + 
            audio_building + 
            cover_sentence(after, audio, sr, all_words, results))


def calculate_coverage(story: str, vocab: set) -> float:
    """
    Calculate the percentage of words in a story that are found in the vocabulary.
    
    Args:
        story: The story text
        vocab: Set of vocabulary words
        
    Returns:
        Coverage percentage as a float
    """
    words = story.split(" ")
    found_words = 0
    
    for word in words:
        clean_word = word.strip().lower().translate(str.maketrans('', '', string.punctuation))
        if clean_word in vocab:
            found_words += 1
            
    return found_words / len(words) if words else 0


def transcribe_audio(audio_path: str, model_name: str = "tiny") -> dict:
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model to use
        
    Returns:
        Transcription results dictionary
    """
    print(f"Loading audio from: {audio_path}")
    sr = 16000
    audio = whisper.load_audio(audio_path, sr=sr)
    
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device="cpu")
    
    print("Transcribing audio...")
    results = whisper.transcribe(model, audio, language="en")
    
    return results, audio, sr


def save_transcription(results: dict, output_path: str) -> None:
    """
    Save transcription results to a JSON file.
    
    Args:
        results: Transcription results
        output_path: Path to save the transcription
    """
    print(f"Saving transcription to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def generate_wordcloud(results: dict, output_path: str) -> None:
    """
    Generate and save a wordcloud from the transcription.
    
    Args:
        results: Transcription results
        output_path: Path to save the wordcloud image
    """
    print(f"Generating wordcloud: {output_path}")
    wc = wordcloud.WordCloud(
        width=2000, 
        height=2000,
        background_color='white',
        stopwords=None,
        min_font_size=10
    ).generate(results["text"])
    
    wc.to_file(output_path)


def generate_story(transcription_text: str) -> str:
    """
    Generate a story from transcription using GPT-5.
    
    Args:
        transcription_text: The transcription text
        
    Returns:
        Generated story text
    """
    print("Generating story from transcription using GPT-5...")
    
    openai_client = OpenAI()
    story_prompt = """
    From a given transcription, generate a short story with as many funny and quirky elements.
    You should only use the words from the transcription, and, as much as possible,
    use entire sentences or long phrases from the transcription.
    The story should be fun and create new meaning from the transcription.
    """
    
    story_response = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": story_prompt},
            {"role": "user", "content": transcription_text},
        ],
    )
    
    generated_story = story_response.choices[0].message.content
    print(f"Generated story:\n{generated_story}")
    
    return generated_story


def process_story_audio(generated_story: str, audio: np.ndarray, sr: int, all_words: List[dict], results: dict) -> np.ndarray:
    """
    Process the generated story into audio using the original audio segments.
    
    Args:
        generated_story: The generated story text
        audio: Original audio array
        sr: Sample rate
        all_words: List of word dictionaries
        results: Whisper transcription results
        
    Returns:
        Audio array for the story
    """
    print("Generating audio for the story...")
    sentences = generated_story.split(". ")
    
    story_audio = []
    for i, sentence in enumerate(sentences, 1):
        if sentence.strip():
            print(f"Processing sentence {i}/{len(sentences)}: {sentence}")
            story_audio.extend(cover_sentence(sentence, audio, sr, all_words, results))
    
    return np.array(story_audio)


def main():
    """Main function to process audio and generate story."""
    parser = argparse.ArgumentParser(
        description="Process audio files for Echoes Installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepareAudio.py input.wav output.wav
  python prepareAudio.py input.wav output.wav --transcription transcript.json
  python prepareAudio.py input.wav output.wav --wordcloud cloud.png
  python prepareAudio.py input.wav output.wav --no-wordcloud
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input audio file"
    )
    
    parser.add_argument(
        "output_file", 
        help="Path for the output audio file"
    )
    
    parser.add_argument(
        "--transcription",
        default="transcription.json",
        help="Path for transcription output (default: transcription.json)"
    )
    
    parser.add_argument(
        "--wordcloud",
        default="wordcloud.png",
        help="Path for wordcloud output (default: wordcloud.png)"
    )
    
    parser.add_argument(
        "--no-wordcloud",
        action="store_true",
        help="Skip wordcloud generation"
    )
    
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: tiny)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1
    
    # Load environment variables
    load_environment()
    
    try:
        # Transcribe audio
        results, audio, sr = transcribe_audio(args.input_file, args.model)
        
        # Save transcription
        save_transcription(results, args.transcription)
        
        # Generate wordcloud if requested
        if not args.no_wordcloud:
            generate_wordcloud(results, args.wordcloud)
        
        # Process words for vocabulary
        all_words = [word for segment in results["segments"] for word in segment["words"]]
        
        # Clean up words (strip, lowercase, remove punctuation)
        for word in all_words:
            word["text"] = word["text"].strip().lower().translate(
                str.maketrans('', '', string.punctuation)
            )
        
        # Build vocabulary set
        vocab = set([word["text"] for word in all_words])
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Sample vocabulary: {random.sample(list(vocab), min(10, len(vocab)))}")
        
        # Generate story
        generated_story = generate_story(results["text"])
        
        # Calculate coverage
        coverage = calculate_coverage(generated_story, vocab)
        print(f"Story coverage: {coverage:.1%}")
        
        # Generate story audio
        story_audio = process_story_audio(generated_story, audio, sr, all_words, results)
        
        # Save the audio
        print(f"Saving story audio to: {args.output_file}")
        sf.write(args.output_file, story_audio, sr)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Output files:")
        print(f"   Audio: {args.output_file}")
        print(f"   Transcription: {args.transcription}")
        if not args.no_wordcloud:
            print(f"   Wordcloud: {args.wordcloud}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
