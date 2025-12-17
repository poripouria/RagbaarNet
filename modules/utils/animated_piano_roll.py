"""
Animated Piano Roll Player for MIDI Files
==========================================

This script creates an animated piano roll visualization synchronized with MIDI playback.
Notes scroll by and are color-coded based on their playback status.

Usage:
    python animated_piano_roll.py path/to/midi_file.mid
    python animated_piano_roll.py path/to/midi_file.mid --duration 45 --fps 30

Author: Generated for Music Dataset Analysis
Date: December 2025
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import mido
import threading
import time
from pathlib import Path


class AnimatedPianoRoll:
    """Class to create and manage animated piano roll visualizations"""
    
    def __init__(self, midi_path, duration=30, fps=20, window_size=5):
        """
        Initialize the animated piano roll
        
        Args:
            midi_path: Path to MIDI file
            duration: Maximum playback duration in seconds
            fps: Frames per second for animation
            window_size: Width of the visible time window in seconds
        """
        self.midi_path = midi_path
        self.duration = duration
        self.fps = fps
        self.window_size = window_size
        self.current_time = 0
        self.notes = []
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.note_bars = []
        self.playhead = None
        self.progress_line = None
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
    def load_midi_data(self):
        """Load and process MIDI file data"""
        print(f"📂 Loading MIDI file: {os.path.basename(self.midi_path)}")
        
        try:
            mid = mido.MidiFile(self.midi_path)
            
            # Collect all note events with timing
            self.notes = []
            for track in mid.tracks:
                track_time = 0
                for msg in track:
                    track_time += msg.time / mid.ticks_per_beat * 0.5
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        self.notes.append({
                            'time': track_time,
                            'note': msg.note,
                            'velocity': msg.velocity,
                            'duration': 0,
                            'active': True
                        })
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        # Find matching note_on and set duration
                        for note in reversed(self.notes):
                            if note['note'] == msg.note and note['active'] and note['duration'] == 0:
                                note['duration'] = track_time - note['time']
                                note['active'] = False
                                break
            
            # Filter notes within duration
            self.notes = [n for n in self.notes if n['time'] <= self.duration]
            
            if not self.notes:
                print("⚠️  No notes found in the specified time range!")
                return False
            
            print(f"✅ Loaded {len(self.notes)} notes")
            print(f"   Duration: {self.duration} seconds")
            print(f"   Note range: {min(n['note'] for n in self.notes)} - {max(n['note'] for n in self.notes)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading MIDI file: {e}")
            return False
    
    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Calculate note range
        min_note = min(n['note'] for n in self.notes)
        max_note = max(n['note'] for n in self.notes)
        
        # Setup piano roll axis
        self.playhead = self.ax1.axvline(x=self.window_size/2, color='red', 
                                        linewidth=3, linestyle='--', label='Now Playing')
        
        self.ax1.set_xlim(0, self.window_size)
        self.ax1.set_ylim(min_note - 2, max_note + 2)
        self.ax1.set_xlabel('Time (seconds)', fontsize=12)
        self.ax1.set_ylabel('MIDI Note Number', fontsize=12)
        self.ax1.set_title(f'Animated Piano Roll: {os.path.basename(self.midi_path)}', 
                          fontsize=14, pad=15)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        # Setup progress axis
        time_bins = np.linspace(0, self.duration, 100)
        note_counts = []
        for i in range(len(time_bins) - 1):
            count = sum(1 for n in self.notes if time_bins[i] <= n['time'] < time_bins[i+1])
            note_counts.append(count)
        
        self.ax2.plot(time_bins[:-1], note_counts, color='darkgreen', linewidth=2)
        self.ax2.fill_between(time_bins[:-1], note_counts, alpha=0.3, color='lightgreen')
        self.progress_line = self.ax2.axvline(x=0, color='red', linewidth=3, linestyle='--')
        self.ax2.set_xlabel('Time (seconds)', fontsize=12)
        self.ax2.set_ylabel('Note Density', fontsize=12)
        self.ax2.set_title('Progress', fontsize=14, pad=15)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(0, self.duration)
        
        plt.tight_layout()
    
    def play_music(self):
        """Play MIDI file in background thread"""
        try:
            pygame.mixer.music.load(self.midi_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"⚠️  Audio playback error: {e}")
    
    def update(self, frame):
        """Update function for animation"""
        # Update current time
        self.current_time = frame / self.fps
        
        # Stop if exceeded duration or music stopped
        if self.current_time >= self.duration or not pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            return []
        
        # Clear previous bars
        for bar in self.note_bars:
            bar.remove()
        self.note_bars.clear()
        
        # Update window view
        window_start = max(0, self.current_time - self.window_size/2)
        window_end = window_start + self.window_size
        self.ax1.set_xlim(window_start, window_end)
        
        # Draw notes in current window
        for note in self.notes:
            note_start = note['time']
            note_end = note_start + max(note['duration'], 0.1)
            
            # Only draw if in current window
            if window_start <= note_end and note_start <= window_end:
                # Color based on whether note is currently playing
                if note_start <= self.current_time <= note_end:
                    color = 'yellow'
                    alpha = 0.9
                    edgecolor = 'red'
                    linewidth = 2
                elif note_start > self.current_time:
                    color = 'lightblue'
                    alpha = 0.6
                    edgecolor = 'blue'
                    linewidth = 1
                else:
                    color = 'gray'
                    alpha = 0.3
                    edgecolor = 'darkgray'
                    linewidth = 1
                
                bar = self.ax1.barh(note['note'], note['duration'], 
                                  left=note_start, height=0.8,
                                  color=color, edgecolor=edgecolor, 
                                  alpha=alpha, linewidth=linewidth)
                self.note_bars.extend(bar)
        
        # Update playhead position
        playhead_pos = self.current_time if self.current_time < self.window_size/2 else self.window_size/2
        self.playhead.set_xdata([window_start + playhead_pos])
        
        # Update progress line
        self.progress_line.set_xdata([self.current_time])
        
        return self.note_bars + [self.playhead, self.progress_line]
    
    def play(self):
        """Start the animated visualization with synchronized audio"""
        if not self.load_midi_data():
            return
        
        self.setup_plot()
        
        # Start music in separate thread
        music_thread = threading.Thread(target=self.play_music)
        music_thread.daemon = True
        music_thread.start()
        
        print("\n" + "="*60)
        print("🎵 Starting Playback")
        print("="*60)
        print("💡 Legend:")
        print("   🟨 Yellow notes (red border) = Currently playing")
        print("   🔵 Blue notes = Coming up next")
        print("   ⬜ Gray notes = Already played")
        print("   🔴 Red line = Current position")
        print("="*60)
        print("\n▶️  Close the window to stop playback\n")
        
        # Create animation
        frames = int(self.duration * self.fps)
        anim = animation.FuncAnimation(
            self.fig, self.update, frames=frames,
            interval=1000/self.fps, blit=True, repeat=False
        )
        
        # Show the animation
        plt.show()
        
        # Clean up
        pygame.mixer.music.stop()
        print("\n✅ Playback complete!")


def main():
    """Main function to run from command line"""
    parser = argparse.ArgumentParser(
        description='Animated Piano Roll Player for MIDI Files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python animated_piano_roll.py song.mid
  python animated_piano_roll.py song.mid --duration 45
  python animated_piano_roll.py song.mid --duration 60 --fps 30 --window 8

Color Legend:
  Yellow (red border) = Currently playing notes
  Blue               = Upcoming notes
  Gray               = Already played notes
  Red line           = Current playback position
        """
    )
    
    parser.add_argument('midi_file', type=str, help='Path to MIDI file')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Maximum playback duration in seconds (default: 30)')
    parser.add_argument('--fps', type=int, default=20,
                       help='Frames per second for animation (default: 20)')
    parser.add_argument('--window', type=int, default=5,
                       help='Width of visible time window in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.midi_file):
        print(f"❌ Error: File not found: {args.midi_file}")
        sys.exit(1)
    
    # Create and run the animated piano roll
    print("\n" + "="*60)
    print("🎹 ANIMATED PIANO ROLL PLAYER")
    print("="*60)
    
    player = AnimatedPianoRoll(
        args.midi_file,
        duration=args.duration,
        fps=args.fps,
        window_size=args.window
    )
    
    try:
        player.play()
    except KeyboardInterrupt:
        print("\n\n⏹️  Playback interrupted by user")
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        pygame.mixer.music.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
