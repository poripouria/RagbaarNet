#!/usr/bin/env python3
"""
Performance Monitor for Video Processor Platform
This script monitors the performance improvements made to the platform
"""

import time
import requests
import json
from collections import deque
import statistics

class PerformanceMonitor:
    def __init__(self, processor_url='http://127.0.0.1:5000'):
        self.processor_url = processor_url
        self.frame_times = deque(maxlen=100)  # Store last 100 frame processing times
        self.update_times = deque(maxlen=100)  # Store last 100 update times
        self.start_time = time.time()
        self.last_frame_counter = 0
        
    def check_processor_status(self):
        """Check if processor is running and get status"""
        try:
            response = requests.get(f"{self.processor_url}/api/status", timeout=2)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except:
            return None
    
    def monitor_performance(self, duration_seconds=60):
        """Monitor performance for specified duration"""
        print(f"🔍 Starting performance monitoring for {duration_seconds} seconds...")
        print(f"📡 Monitoring processor at: {self.processor_url}")
        
        # Check if processor is running
        status = self.check_processor_status()
        if not status:
            print("❌ Processor is not running or not accessible!")
            print("   Please start the processor with: python processor.py")
            return
        
        print(f"✅ Processor is running")
        print(f"📊 Initial stats:")
        print(f"   - Frame counter: {status.get('frame_counter', 0)}")
        print(f"   - Processing interval: {status.get('processing_interval', 'unknown')} frames")
        print(f"   - Queue size: {status.get('queue_size', 0)}")
        print()
        
        end_time = time.time() + duration_seconds
        last_status_time = 0
        frame_rates = []
        
        while time.time() < end_time:
            current_time = time.time()
            
            # Get status every second
            if current_time - last_status_time >= 1.0:
                start_request = time.time()
                status = self.check_processor_status()
                request_time = time.time() - start_request
                
                if status:
                    self.update_times.append(request_time * 1000)  # Convert to ms
                    
                    # Calculate frame rate
                    current_frame = status.get('frame_counter', 0)
                    if self.last_frame_counter > 0:
                        frames_processed = current_frame - self.last_frame_counter
                        time_elapsed = current_time - last_status_time
                        frame_rate = frames_processed / time_elapsed if time_elapsed > 0 else 0
                        frame_rates.append(frame_rate)
                        
                        # Real-time display
                        elapsed = current_time - self.start_time
                        print(f"\r⏱️  {elapsed:6.1f}s | "
                              f"📊 Frame: {current_frame:4d} | "
                              f"🚀 FPS: {frame_rate:5.1f} | "
                              f"📤 Queue: {status.get('queue_size', 0):2d} | "
                              f"⚡ API: {request_time*1000:5.1f}ms", end='', flush=True)
                    
                    self.last_frame_counter = current_frame
                    last_status_time = current_time
                else:
                    print(f"\r❌ Connection lost at {current_time - self.start_time:.1f}s", end='', flush=True)
            
            time.sleep(0.1)  # Small delay to prevent overwhelming
        
        print("\n\n📈 Performance Summary:")
        self.print_performance_summary(frame_rates)
    
    def print_performance_summary(self, frame_rates):
        """Print detailed performance summary"""
        if not frame_rates:
            print("❌ No frame rate data collected")
            return
        
        # Frame rate statistics
        avg_fps = statistics.mean(frame_rates)
        max_fps = max(frame_rates)
        min_fps = min(frame_rates)
        fps_std = statistics.stdev(frame_rates) if len(frame_rates) > 1 else 0
        
        print(f"🎥 Frame Rate Performance:")
        print(f"   - Average FPS: {avg_fps:.2f}")
        print(f"   - Maximum FPS: {max_fps:.2f}")
        print(f"   - Minimum FPS: {min_fps:.2f}")
        print(f"   - Std Deviation: {fps_std:.2f}")
        print(f"   - Samples: {len(frame_rates)}")
        
        # API response time statistics
        if self.update_times:
            avg_api = statistics.mean(self.update_times)
            max_api = max(self.update_times)
            min_api = min(self.update_times)
            api_std = statistics.stdev(self.update_times) if len(self.update_times) > 1 else 0
            
            print(f"\n📡 API Response Performance:")
            print(f"   - Average response: {avg_api:.1f}ms")
            print(f"   - Maximum response: {max_api:.1f}ms")
            print(f"   - Minimum response: {min_api:.1f}ms")
            print(f"   - Std Deviation: {api_std:.1f}ms")
            print(f"   - Samples: {len(self.update_times)}")
        
        # Performance evaluation
        print(f"\n⚡ Performance Evaluation:")
        if avg_fps >= 5.0:
            print("   ✅ EXCELLENT: Smooth real-time performance")
        elif avg_fps >= 3.0:
            print("   ✅ GOOD: Acceptable real-time performance")
        elif avg_fps >= 1.5:
            print("   ⚠️  FAIR: Usable but not smooth")
        else:
            print("   ❌ POOR: Performance needs improvement")
        
        if self.update_times and statistics.mean(self.update_times) < 50:
            print("   ✅ Low latency: API responses are fast")
        elif self.update_times and statistics.mean(self.update_times) < 100:
            print("   ⚠️  Medium latency: API responses are acceptable")
        else:
            print("   ❌ High latency: API responses are slow")
    
    def toggle_debug_mode(self, enable=True):
        """Toggle debug mode on the processor"""
        action = 'enable' if enable else 'disable'
        try:
            response = requests.post(f"{self.processor_url}/api/debug/{action}", timeout=2)
            if response.status_code == 200:
                print(f"✅ Debug mode {'enabled' if enable else 'disabled'}")
                return True
            else:
                print(f"❌ Failed to toggle debug mode: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error toggling debug mode: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Video Processor Performance')
    parser.add_argument('--url', default='http://127.0.0.1:5000', help='Processor URL')
    parser.add_argument('--duration', type=int, default=30, help='Monitoring duration in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode during monitoring')
    parser.add_argument('--disable-debug', action='store_true', help='Disable debug mode at start')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.url)
    
    # Toggle debug mode if requested
    if args.disable_debug:
        monitor.toggle_debug_mode(False)
    elif args.debug:
        monitor.toggle_debug_mode(True)
    
    # Run monitoring
    try:
        monitor.monitor_performance(args.duration)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
    
    # Disable debug mode after monitoring if it was enabled
    if args.debug:
        monitor.toggle_debug_mode(False)

if __name__ == '__main__':
    main()
