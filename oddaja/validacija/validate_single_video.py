#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_json(file_path: str) -> Dict:
    """Load JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        sys.exit(1)

def convert_to_frame_events(annotations: List[Dict]) -> Dict[int, List[str]]:
    """
    Convert frame range annotations to per-frame events.
    Input format: [{"frame_start": X, "frame_stop": Y, "event": ["event1", "event2"]}, ...]
    Output format: {frame_number: ["event1", "event2"], ...}
    """
    frame_events = {}
    
    for annotation in annotations:
        frame_start = annotation["frame_start"]
        frame_stop = annotation["frame_stop"]
        events = annotation["event"]
        
        for frame in range(frame_start, frame_stop + 1):
            frame_events[frame] = events
            
    return frame_events

def create_confusion_matrix(ground_truth: Dict[int, List[str]], student_output: Dict[str, List[str]], events: List[str]) -> np.ndarray:
    """Create confusion matrix from ground truth and student output."""
    # Convert student output frame numbers to integers
    student_output_int = {int(k): v for k, v in student_output.items()}
    
    # Initialize confusion matrix
    n_events = len(events)
    conf_matrix = np.zeros((n_events, n_events), dtype=int)
    
    # Create event to index mapping
    event_to_idx = {event: idx for idx, event in enumerate(events)}
    
    # Fill confusion matrix
    for frame_id, gt_events in ground_truth.items():
        if frame_id not in student_output_int:
            continue
            
        gt_event = gt_events[0]  # We know there's only one event per frame
        pred_event = student_output_int[frame_id][0]  # We know there's only one event per frame
        
        gt_idx = event_to_idx[gt_event]
        pred_idx = event_to_idx[pred_event]
        
        conf_matrix[gt_idx][pred_idx] += 1
    
    return conf_matrix

def plot_confusion_matrix(conf_matrix: np.ndarray, events: List[str], output_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=events,
                yticklabels=events)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Event')
    plt.ylabel('True Event')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def calculate_metrics(ground_truth: Dict[int, List[str]], student_output: Dict[str, List[str]]) -> Dict:
    """
    Calculate overall metrics between ground truth and student output.
    """
    # Convert student output frame numbers to integers
    student_output_int = {int(k): v for k, v in student_output.items()}
    
    metrics = {
        "total_frames": len(ground_truth),
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "missing_predictions": 0,
        "extra_predictions": 0
    }
    
    # Compare each frame's events
    for frame_id, gt_events in ground_truth.items():
        if frame_id not in student_output_int:
            metrics["missing_predictions"] += 1
            continue
            
        student_events = student_output_int[frame_id]
        
        # Compare events for this frame
        if set(gt_events) == set(student_events):
            metrics["correct_predictions"] += 1
        else:
            metrics["incorrect_predictions"] += 1
            
    # Count extra predictions (frames in student output not in ground truth)
    metrics["extra_predictions"] = len(student_output_int) - len(ground_truth)
    
    # Calculate overall accuracy
    total_predictions = metrics["correct_predictions"] + metrics["incorrect_predictions"]
    if total_predictions > 0:
        metrics["accuracy"] = metrics["correct_predictions"] / total_predictions
    else:
        metrics["accuracy"] = 0.0
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Validate student video processing algorithm')
    parser.add_argument('video_path', type=str, help='Path to the first input video file')
    parser.add_argument('video_path_2', type=str, help='Path to the second input video file')
    parser.add_argument('ground_truth_path', type=str, help='Path to the ground truth JSON file')
    parser.add_argument('student_output_path', type=str, help='Path to the student output JSON file')
    args = parser.parse_args()
    
    # Load ground truth annotations
    ground_truth_data = load_json(args.ground_truth_path)
    ground_truth = convert_to_frame_events(ground_truth_data["annotations"])
    
    """
    YOUR CODE HERE. Tukaj vključite klicanje vaše funkcije, ki obdela video in vrne/shrani json z rezultati.
    """
    
    # Load student output
    student_output = load_json(args.student_output_path)
    
    # Get unique events for confusion matrix
    all_events = sorted(set(event for events in ground_truth.values() for event in events))
    
    # Create and save confusion matrix
    conf_matrix = create_confusion_matrix(ground_truth, student_output, all_events)
    output_path = Path(args.student_output_path).stem + "_confusion_matrix.png"
    plot_confusion_matrix(conf_matrix, all_events, output_path)
    print(f"\nConfusion matrix saved to: {output_path}")
    
    # Calculate and display metrics
    metrics = calculate_metrics(ground_truth, student_output)
    
    print("\nValidation Results:")
    print("-" * 50)
    print(f"Total frames processed: {metrics['total_frames']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Incorrect predictions: {metrics['incorrect_predictions']}")
    print(f"Missing predictions: {metrics['missing_predictions']}")
    print(f"Extra predictions: {metrics['extra_predictions']}")
    print(f"Overall accuracy: {metrics['accuracy']:.2%}")
    print("-" * 50)

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# NAVODILA ZA UPORABO:
# Skripto za validacijo kličeš iz terminala na sledeč način:
#
#   python validate_single_video.py <pot_do_videa> <pot_do_ground_truth_json> <pot_do_student_output_json>
#
# Primer:
#   python validate_single_video.py 64210054_video_15.mp4 64210054_video_15.json 64210054_video_15_predictions.json
#
# Skripta bo izpisala metrike in shranila confusion matrix kot PNG sliko.
# ------------------------------------------------------------
    