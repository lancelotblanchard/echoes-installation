#!/bin/bash

# Audio Concatenation Script
# 
# This script processes and concatenates multiple WAV files by:
# 1. Normalizing audio levels
# 2. Converting to mono
# 3. Concatenating all files into a single output file
#
# Usage: ./script.sh [input_directory] [output_file]
# Examples:
#   ./script.sh ./audio_files output.wav
#   ./script.sh . final_audio.wav
#   ./script.sh  # Uses current directory and output.wav

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Audio Concatenation Script

Usage: $0 [OPTIONS] [INPUT_DIR] [OUTPUT_FILE]

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -d, --dry-run       Show what would be done without executing

Arguments:
    INPUT_DIR           Directory containing WAV files (default: current directory)
    OUTPUT_FILE         Output file name (default: output.wav)

Examples:
    $0                           # Process current directory, output to output.wav
    $0 ./audio_files            # Process ./audio_files, output to output.wav
    $0 ./audio_files final.wav  # Process ./audio_files, output to final.wav
    $0 -v . final_audio.wav     # Verbose processing

EOF
}

# Function to check dependencies
check_dependencies() {
    if ! command -v ffmpeg &> /dev/null; then
        print_error "ffmpeg is not installed. Please install ffmpeg first."
        exit 1
    fi
}

# Function to validate input directory
validate_input_dir() {
    local dir="$1"
    
    if [[ ! -d "$dir" ]]; then
        print_error "Input directory '$dir' does not exist"
        exit 1
    fi
    
    # Check if directory contains WAV files
    local wav_count=$(find "$dir" -maxdepth 1 -name "*.wav" | wc -l)
    if [[ $wav_count -eq 0 ]]; then
        print_error "No WAV files found in directory '$dir'"
        exit 1
    fi
    
    print_status "Found $wav_count WAV file(s) in '$dir'"
}

# Function to process audio files
process_audio_files() {
    local input_dir="$1"
    local temp_dir="$2"
    local verbose="$3"
    
    local processed_count=0
    local total_files=$(find "$input_dir" -maxdepth 1 -name "*.wav" | wc -l)
    
    print_status "Processing $total_files audio file(s)..."
    
    # Create file list
    > filelist.txt
    
    for f in "$input_dir"/*.wav; do
        # Skip if no files match pattern
        [[ -f "$f" ]] || continue
        
        local basename=$(basename "$f")
        local output_path="$temp_dir/$basename"
        
        print_status "Processing: $basename ($((++processed_count))/$total_files)"
        
        if [[ "$verbose" == "true" ]]; then
            ffmpeg -i "$f" -af "loudnorm,pan=mono|c0=c0" "$output_path"
        else
            ffmpeg -i "$f" -af "loudnorm,pan=mono|c0=c0" "$output_path" -loglevel error
        fi
        
        echo "file '$output_path'" >> filelist.txt
    done
    
    print_success "Processed $processed_count file(s)"
}

# Function to concatenate files
concatenate_files() {
    local output_file="$1"
    local verbose="$2"
    
    print_status "Concatenating processed files..."
    
    if [[ "$verbose" == "true" ]]; then
        ffmpeg -f concat -safe 0 -i filelist.txt -c copy "$output_file"
    else
        ffmpeg -f concat -safe 0 -i filelist.txt -c copy "$output_file" -loglevel error
    fi
    
    print_success "Concatenation complete: $output_file"
}

# Function to cleanup
cleanup() {
    local temp_dir="$1"
    
    print_status "Cleaning up temporary files..."
    rm -rf "$temp_dir" filelist.txt
    print_success "Cleanup complete"
}

# Main function
main() {
    local verbose="false"
    local dry_run="false"
    local input_dir="."
    local output_file="output.wav"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                verbose="true"
                shift
                ;;
            -d|--dry-run)
                dry_run="true"
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$input_dir" || "$input_dir" == "." ]]; then
                    input_dir="$1"
                elif [[ -z "$output_file" || "$output_file" == "output.wav" ]]; then
                    output_file="$1"
                else
                    print_error "Too many arguments"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate output file extension
    if [[ ! "$output_file" =~ \.wav$ ]]; then
        output_file="${output_file}.wav"
        print_warning "Output file extension changed to .wav: $output_file"
    fi
    
    # Check if output file already exists
    if [[ -f "$output_file" ]]; then
        print_warning "Output file '$output_file' already exists"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Operation cancelled"
            exit 0
        fi
    fi
    
    if [[ "$dry_run" == "true" ]]; then
        print_status "DRY RUN - Would process:"
        print_status "  Input directory: $input_dir"
        print_status "  Output file: $output_file"
        print_status "  Verbose: $verbose"
        exit 0
    fi
    
    # Check dependencies
    check_dependencies
    
    # Validate input directory
    validate_input_dir "$input_dir"
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    print_status "Created temporary directory: $temp_dir"
    
    # Set up cleanup trap
    trap 'cleanup "$temp_dir"' EXIT
    
    # Process audio files
    process_audio_files "$input_dir" "$temp_dir" "$verbose"
    
    # Concatenate files
    concatenate_files "$output_file" "$verbose"
    
    # Get final file size
    local file_size=$(du -h "$output_file" | cut -f1)
    print_success "Final output: $output_file ($file_size)"
}

# Run main function with all arguments
main "$@"
