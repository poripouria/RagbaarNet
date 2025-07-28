import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import json
import os
from datetime import datetime

class InteractiveMusicPlatform:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Music Generation Platform")
        self.root.configure(bg='#2b2b2b')
        
        # Get screen dimensions and set window to landscape
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.8)
        
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(True, True)
        
        # Variables
        self.current_frame = None
        self.video_capture = None
        self.is_recording = False
        self.input_source = None
        self.roi_points = [[100, 100], [500, 100], [500, 400], [100, 400]]  # Default ROI points
        self.dragging_point = None
        self.settings = self.load_settings()
        
        # Colors
        self.bg_color = '#2b2b2b'
        self.menu_color = '#1e1e1e'
        self.button_color = '#4a4a4a'
        self.accent_color = '#00ff88'
        self.text_color = '#ffffff'
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Create main container
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Check if input source is selected
        if not self.input_source:
            self.show_input_selection()
        else:
            self.setup_main_interface(main_container)
    
    def show_input_selection(self):
        """Show input source selection dialog"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Input Source")
        selection_window.geometry("400x300")
        selection_window.configure(bg=self.bg_color)
        selection_window.transient(self.root)
        selection_window.grab_set()
        
        # Center the window
        selection_window.geometry("+{}+{}".format(
            int(self.root.winfo_screenwidth()/2 - 200),
            int(self.root.winfo_screenheight()/2 - 150)
        ))
        
        # Title
        title_label = tk.Label(selection_window, text="Choose Input Source", 
                              font=("Arial", 16, "bold"), fg=self.text_color, bg=self.bg_color)
        title_label.pack(pady=20)
        
        # Input source buttons
        button_frame = tk.Frame(selection_window, bg=self.bg_color)
        button_frame.pack(expand=True, fill=tk.BOTH, padx=20)
        
        buttons = [
            ("📱 Phone Camera", lambda: self.select_input_source("phone_camera", selection_window)),
            ("🎥 Video File", lambda: self.select_input_source("video_file", selection_window)),
            ("📱 Screen Record", lambda: self.select_input_source("screen_record", selection_window)),
            ("🌐 Network Stream", lambda: self.select_input_source("network_stream", selection_window))
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = tk.Button(button_frame, text=text, command=command,
                           font=("Arial", 12), bg=self.button_color, fg=self.text_color,
                           relief=tk.FLAT, pady=10, cursor="hand2")
            btn.pack(fill=tk.X, pady=5)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.accent_color))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.button_color))
    
    def select_input_source(self, source_type, window):
        """Handle input source selection"""
        self.input_source = source_type
        window.destroy()
        
        if source_type == "video_file":
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            if file_path:
                self.video_path = file_path
                self.setup_main_interface_after_selection()
            else:
                self.input_source = None
                self.show_input_selection()
        elif source_type == "network_stream":
            self.show_url_input()
        else:
            self.setup_main_interface_after_selection()
    
    def show_url_input(self):
        """Show URL input dialog for network stream"""
        url_window = tk.Toplevel(self.root)
        url_window.title("Enter Stream URL")
        url_window.geometry("400x150")
        url_window.configure(bg=self.bg_color)
        url_window.transient(self.root)
        url_window.grab_set()
        
        tk.Label(url_window, text="Enter DashCam Stream URL:", 
                font=("Arial", 12), fg=self.text_color, bg=self.bg_color).pack(pady=10)
        
        url_entry = tk.Entry(url_window, font=("Arial", 10), width=50)
        url_entry.pack(pady=10)
        url_entry.insert(0, "http://192.168.1.100:8080/video")  # Default example
        
        button_frame = tk.Frame(url_window, bg=self.bg_color)
        button_frame.pack(pady=10)
        
        def confirm_url():
            url = url_entry.get().strip()
            if url:
                self.stream_url = url
                url_window.destroy()
                self.setup_main_interface_after_selection()
            else:
                messagebox.showerror("Error", "Please enter a valid URL")
        
        def cancel_url():
            self.input_source = None
            url_window.destroy()
            self.show_input_selection()
        
        tk.Button(button_frame, text="Confirm", command=confirm_url,
                 bg=self.accent_color, fg="black", relief=tk.FLAT, padx=20).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_url,
                 bg=self.button_color, fg=self.text_color, relief=tk.FLAT, padx=20).pack(side=tk.LEFT, padx=5)
    
    def setup_main_interface_after_selection(self):
        """Setup main interface after input source selection"""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Create main container
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        self.setup_main_interface(main_container)
    
    def setup_main_interface(self, parent):
        """Setup the main interface with menu (1/5) and display area (4/5)"""
        # Create horizontal layout
        # Menu panel (1/5 of width)
        menu_width = int(self.root.winfo_width() * 0.2) if self.root.winfo_width() > 1 else 250
        
        self.menu_frame = tk.Frame(parent, bg=self.menu_color, width=menu_width)
        self.menu_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2))
        self.menu_frame.pack_propagate(False)
        
        # Display area (4/5 of width)
        self.display_frame = tk.Frame(parent, bg=self.bg_color)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_menu()
        self.setup_display_area()
        
        # Start video capture
        self.start_video_capture()
    
    def setup_menu(self):
        """Setup the side menu"""
        # Title
        title_label = tk.Label(self.menu_frame, text="AI Music\nPlatform", 
                              font=("Arial", 14, "bold"), fg=self.accent_color, bg=self.menu_color)
        title_label.pack(pady=(20, 30))
        
        # Input source info
        source_info = {
            "phone_camera": "📱 Phone Camera",
            "video_file": "🎥 Video File", 
            "screen_record": "📱 Screen Record",
            "network_stream": "🌐 Network Stream"
        }
        
        if self.input_source in source_info:
            tk.Label(self.menu_frame, text=f"Source:\n{source_info[self.input_source]}", 
                    font=("Arial", 10), fg=self.text_color, bg=self.menu_color).pack(pady=(0, 20))
        
        # Menu buttons
        menu_buttons = [
            ("🔄 Change Source", self.change_input_source),
            ("⚙️ ROI Settings", self.toggle_roi_editing),
            ("💾 Save Settings", self.save_current_settings),
            ("📁 Load Settings", self.load_settings_file),
            ("🎵 Start Music Gen", self.start_music_generation),
            ("⏸️ Pause/Resume", self.toggle_pause),
            ("📸 Screenshot", self.take_screenshot),
            ("🔙 Back", self.go_back)
        ]
        
        for text, command in menu_buttons:
            btn = tk.Button(self.menu_frame, text=text, command=command,
                           font=("Arial", 10), bg=self.button_color, fg=self.text_color,
                           relief=tk.FLAT, pady=8, width=18, cursor="hand2")
            btn.pack(pady=2, padx=10, fill=tk.X)
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.accent_color, fg="black"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.button_color, fg=self.text_color))
        
        # Status section
        tk.Label(self.menu_frame, text="Status:", font=("Arial", 10, "bold"), 
                fg=self.text_color, bg=self.menu_color).pack(pady=(30, 5))
        
        self.status_label = tk.Label(self.menu_frame, text="Ready", font=("Arial", 9), 
                                   fg=self.accent_color, bg=self.menu_color, wraplength=200)
        self.status_label.pack(pady=(0, 10))
        
        # ROI Info
        tk.Label(self.menu_frame, text="ROI Points:", font=("Arial", 10, "bold"), 
                fg=self.text_color, bg=self.menu_color).pack(pady=(10, 5))
        self.roi_info_label = tk.Label(self.menu_frame, text="", font=("Arial", 8), 
                                     fg=self.text_color, bg=self.menu_color, wraplength=200)
        self.roi_info_label.pack()
    
    def setup_display_area(self):
        """Setup the main display area for video/camera feed"""
        # Video display canvas
        self.canvas = tk.Canvas(self.display_frame, bg='black', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind mouse events for ROI point manipulation
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        
        # Instructions
        instructions = tk.Label(self.display_frame, 
                              text="Drag the green points to adjust ROI (Region of Interest)", 
                              font=("Arial", 10), fg=self.accent_color, bg=self.bg_color)
        instructions.pack(pady=(0, 5))
    
    def start_video_capture(self):
        """Start video capture based on selected input source"""
        try:
            if self.input_source == "phone_camera":
                self.video_capture = cv2.VideoCapture(0)  # Default camera
            elif self.input_source == "video_file":
                self.video_capture = cv2.VideoCapture(self.video_path)
            elif self.input_source == "screen_record":
                # For screen recording, we'll use a placeholder or implement screen capture
                self.video_capture = cv2.VideoCapture(0)  # Fallback to camera
            elif self.input_source == "network_stream":
                self.video_capture = cv2.VideoCapture(self.stream_url)
            
            if self.video_capture and self.video_capture.isOpened():
                self.update_status("Connected - Receiving video feed")
                self.update_video_feed()
            else:
                self.update_status("Error: Could not connect to video source")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
    
    def update_video_feed(self):
        """Update video feed in the canvas"""
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                self.display_frame_with_roi(frame)
                self.update_roi_info()
        
        # Schedule next update
        self.root.after(33, self.update_video_feed)  # ~30 FPS
    
    def display_frame_with_roi(self, frame):
        """Display frame with ROI overlay"""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, lambda: self.display_frame_with_roi(frame))
            return
        
        # Resize frame to fit canvas while maintaining aspect ratio
        frame_height, frame_width = frame.shape[:2]
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB and then to PhotoImage
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        
        # Center the image
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep a reference
        
        # Store scaling factors for ROI point mapping
        self.scale_x = new_width / frame_width
        self.scale_y = new_height / frame_height
        self.offset_x = x_offset
        self.offset_y = y_offset
        
        # Draw ROI
        self.draw_roi()
    
    def draw_roi(self):
        """Draw ROI points and convex hull"""
        if not hasattr(self, 'scale_x'):
            return
        
        # Convert ROI points to canvas coordinates
        canvas_points = []
        for point in self.roi_points:
            x = int(point[0] * self.scale_x + self.offset_x)
            y = int(point[1] * self.scale_y + self.offset_y)
            canvas_points.append([x, y])
        
        # Draw convex hull (polygon)
        if len(canvas_points) >= 3:
            flat_points = [coord for point in canvas_points for coord in point]
            self.canvas.create_polygon(flat_points, fill="", outline=self.accent_color, 
                                     width=2, tags="roi_polygon")
        
        # Draw ROI points
        for i, point in enumerate(canvas_points):
            x, y = point
            self.canvas.create_oval(x-8, y-8, x+8, y+8, fill=self.accent_color, 
                                  outline="white", width=2, tags=f"roi_point_{i}")
            self.canvas.create_text(x, y-15, text=str(i+1), fill="white", 
                                  font=("Arial", 10, "bold"), tags=f"roi_label_{i}")
    
    def on_canvas_click(self, event):
        """Handle canvas click for ROI point selection"""
        # Check if click is near any ROI point
        for i, point in enumerate(self.roi_points):
            canvas_x = int(point[0] * self.scale_x + self.offset_x)
            canvas_y = int(point[1] * self.scale_y + self.offset_y)
            
            if abs(event.x - canvas_x) < 12 and abs(event.y - canvas_y) < 12:
                self.dragging_point = i
                self.canvas.configure(cursor="hand2")
                break
    
    def on_canvas_drag(self, event):
        """Handle canvas drag for ROI point movement"""
        if self.dragging_point is not None and hasattr(self, 'scale_x'):
            # Convert canvas coordinates back to frame coordinates
            frame_x = (event.x - self.offset_x) / self.scale_x
            frame_y = (event.y - self.offset_y) / self.scale_y
            
            # Clamp to frame boundaries
            frame_x = max(0, min(frame_x, self.current_frame.shape[1] if self.current_frame is not None else 640))
            frame_y = max(0, min(frame_y, self.current_frame.shape[0] if self.current_frame is not None else 480))
            
            self.roi_points[self.dragging_point] = [frame_x, frame_y]
    
    def on_canvas_release(self, event):
        """Handle canvas mouse release"""
        self.dragging_point = None
        self.canvas.configure(cursor="crosshair")
    
    def on_canvas_motion(self, event):
        """Handle canvas mouse motion for cursor changes"""
        if hasattr(self, 'scale_x'):
            cursor = "crosshair"
            for i, point in enumerate(self.roi_points):
                canvas_x = int(point[0] * self.scale_x + self.offset_x)
                canvas_y = int(point[1] * self.scale_y + self.offset_y)
                
                if abs(event.x - canvas_x) < 12 and abs(event.y - canvas_y) < 12:
                    cursor = "hand2"
                    break
            
            self.canvas.configure(cursor=cursor)
    
    def update_roi_info(self):
        """Update ROI information in the menu"""
        roi_text = ""
        for i, point in enumerate(self.roi_points):
            roi_text += f"P{i+1}: ({int(point[0])}, {int(point[1])})\n"
        self.roi_info_label.configure(text=roi_text)
    
    def update_status(self, message):
        """Update status message"""
        self.status_label.configure(text=message)
    
    # Menu button callbacks
    def change_input_source(self):
        """Change input source"""
        if self.video_capture:
            self.video_capture.release()
        self.input_source = None
        self.show_input_selection()
    
    def toggle_roi_editing(self):
        """Toggle ROI editing mode"""
        messagebox.showinfo("ROI Settings", "Drag the green points to adjust the Region of Interest")
    
    def save_current_settings(self):
        """Save current settings"""
        settings = {
            "input_source": self.input_source,
            "roi_points": self.roi_points,
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(self, 'video_path'):
            settings["video_path"] = self.video_path
        if hasattr(self, 'stream_url'):
            settings["stream_url"] = self.stream_url
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Settings"
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
            self.update_status("Settings saved successfully")
    
    def load_settings_file(self):
        """Load settings from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Settings"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                
                if "roi_points" in settings:
                    self.roi_points = settings["roi_points"]
                
                self.update_status("Settings loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load settings: {str(e)}")
    
    def load_settings(self):
        """Load default settings"""
        settings_file = "platform_settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def start_music_generation(self):
        """Start music generation (placeholder)"""
        messagebox.showinfo("Music Generation", 
                           "Music generation will be implemented here!\n"
                           f"ROI Points: {self.roi_points}\n"
                           f"Input Source: {self.input_source}")
        self.update_status("Music generation started")
    
    def toggle_pause(self):
        """Toggle pause/resume"""
        self.update_status("Paused" if not hasattr(self, 'paused') or not self.paused else "Resumed")
        self.paused = not (hasattr(self, 'paused') and self.paused)
    
    def take_screenshot(self):
        """Take screenshot"""
        if self.current_frame is not None:
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.update_status(f"Screenshot saved: {filename}")
    
    def go_back(self):
        """Go back to input selection"""
        self.change_input_source()
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    app = InteractiveMusicPlatform()
    app.run()