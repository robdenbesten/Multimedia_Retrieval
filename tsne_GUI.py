import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import numpy as np

class TSNEVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("t-SNE Data Visualizer")
        self.root.geometry("1200x800")
        
        # Load data
        self.load_data()
        
        # Initialize plot variables
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.category_colors = {}
        self.category_visibility = {}
        self.scatter_plots = {}
        
        # Initialize hover functionality
        self.hover_annotation = None
        self.visible_data_points = []  # Store visible data points for hover detection
        self.category_scatter_plots = {}  # Store scatter plot objects for each category
        self.currently_highlighted_category = None
        
        # Calculate fixed axis limits based on full dataset
        self.x_min = self.data['x'].min()
        self.x_max = self.data['x'].max()
        self.y_min = self.data['y'].min()
        self.y_max = self.data['y'].max()
        
        # Add some padding to the limits
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        padding = 0.05  # 5% padding
        
        self.x_min -= x_range * padding
        self.x_max += x_range * padding
        self.y_min -= y_range * padding
        self.y_max += y_range * padding
        
        # Create GUI elements
        self.create_widgets()
        
        # Initial plot
        self.update_plot()
    
    def load_data(self):
        """Load the t-SNE data from CSV file"""
        try:
            self.data = pd.read_csv('tsne_plot_data.csv')
            self.categories = sorted(self.data['category'].unique())
            print(f"Loaded {len(self.data)} data points with {len(self.categories)} categories")
        except FileNotFoundError:
            tk.messagebox.showerror("Error", "tsne_plot_data.csv not found!")
            self.root.destroy()
            return
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.root.destroy()
            return
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Category Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect hover event
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        # Add toolbar for matplotlib
        toolbar = tk.Frame(plot_frame)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar)
        nav_toolbar.update()
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Show All", command=self.show_all_categories).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Hide All", command=self.hide_all_categories).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Refresh", command=self.update_plot).pack(side=tk.LEFT)
        
        # Create scrollable frame for checkboxes
        self.create_category_checkboxes(control_frame)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT)
        self.stats_label.pack()
        
        self.update_statistics()
    
    def create_category_checkboxes(self, parent):
        """Create checkboxes for each category with scrollbar"""
        # Frame for scrollable area
        scroll_frame = ttk.Frame(parent)
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(scroll_frame, height=300)
        scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate colors for categories
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.categories)))
        if len(self.categories) > 20:
            colors = plt.cm.hsv(np.linspace(0, 1, len(self.categories)))
        
        # Create checkboxes for each category
        self.category_vars = {}
        for i, category in enumerate(self.categories):
            # Assign color
            self.category_colors[category] = colors[i]
            
            # Create checkbox variable
            var = tk.BooleanVar(value=True)
            self.category_vars[category] = var
            self.category_visibility[category] = True
            
            # Create checkbox with color indicator
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=1)
            
            # Color square
            color_hex = mcolors.to_hex(colors[i])
            color_label = tk.Label(frame, width=3, height=1, bg=color_hex, relief=tk.RAISED)
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Checkbox
            cb = ttk.Checkbutton(
                frame, 
                text=f"{category} ({len(self.data[self.data['category'] == category])})",
                variable=var,
                command=self.on_category_toggle
            )
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def on_category_toggle(self):
        """Handle category checkbox toggle"""
        for category, var in self.category_vars.items():
            self.category_visibility[category] = var.get()
        self.update_plot()
        self.update_statistics()
    
    def show_all_categories(self):
        """Show all categories"""
        for var in self.category_vars.values():
            var.set(True)
        for category in self.categories:
            self.category_visibility[category] = True
        self.update_plot()
        self.update_statistics()
    
    def hide_all_categories(self):
        """Hide all categories"""
        for var in self.category_vars.values():
            var.set(False)
        for category in self.categories:
            self.category_visibility[category] = False
        self.update_plot()
        self.update_statistics()
    
    def update_plot(self):
        """Update the scatter plot based on visible categories"""
        self.ax.clear()
        
        visible_points = 0
        self.visible_data_points = []  # Reset visible data points for hover detection
        self.category_scatter_plots = {}  # Reset scatter plot objects
        self.currently_highlighted_category = None
        
        for category in self.categories:
            if self.category_visibility.get(category, True):
                category_data = self.data[self.data['category'] == category]
                
                scatter = self.ax.scatter(
                    category_data['x'], 
                    category_data['y'],
                    c=[self.category_colors[category]], 
                    label=category,
                    alpha=1.0,
                    s=20
                )
                
                # Store scatter plot object for this category
                self.category_scatter_plots[category] = scatter
                
                # Store data for hover functionality
                for _, row in category_data.iterrows():
                    self.visible_data_points.append({
                        'x': row['x'],
                        'y': row['y'],
                        'category': row['category'],
                        'object_name': row['object_name']
                    })
                
                visible_points += len(category_data)
        
        # Set fixed axis limits to keep the view consistent
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        
        self.ax.set_xlabel('t-SNE Dimension 1')
        self.ax.set_ylabel('t-SNE Dimension 2')
        self.ax.set_title(f't-SNE Visualization of 3D Shape Database\n({visible_points} points visible)')
        self.ax.grid(True, alpha=0.3)
        
        # Only show legend if there are visible categories and not too many
        visible_categories = sum(1 for v in self.category_visibility.values() if v)
        if visible_categories > 0 and visible_categories <= 15:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Create hover annotation (initially invisible)
        self.hover_annotation = self.ax.annotate('', xy=(0,0), xytext=(20,20), 
                                               textcoords="offset points",
                                               bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                                               arrowprops=dict(arrowstyle="->"))
        self.hover_annotation.set_visible(False)
        
        self.canvas.draw()
    
    def update_statistics(self):
        """Update statistics display"""
        visible_categories = [cat for cat, visible in self.category_visibility.items() if visible]
        visible_points = sum(len(self.data[self.data['category'] == cat]) for cat in visible_categories)
        
        stats_text = f"Total categories: {len(self.categories)}\n"
        stats_text += f"Visible categories: {len(visible_categories)}\n"
        stats_text += f"Total points: {len(self.data)}\n"
        stats_text += f"Visible points: {visible_points}"
        
        self.stats_label.config(text=stats_text)
    
    def highlight_category(self, target_category):
        """Highlight a specific category by making others more transparent"""
        if target_category == self.currently_highlighted_category:
            return  # Already highlighting this category
        
        self.currently_highlighted_category = target_category
        
        for category, scatter_plot in self.category_scatter_plots.items():
            if category == target_category:
                # Keep the target category at full opacity (100%)
                scatter_plot.set_alpha(1.0)
            else:
                # Make other categories 1% opacity
                scatter_plot.set_alpha(0.01)
        
        self.canvas.draw_idle()
    
    def reset_category_highlighting(self):
        """Reset all categories to normal opacity"""
        if self.currently_highlighted_category is None:
            return  # Nothing to reset
        
        self.currently_highlighted_category = None
        
        for scatter_plot in self.category_scatter_plots.values():
            scatter_plot.set_alpha(1.0)  # Reset to full opacity
        
        self.canvas.draw_idle()
    
    def on_hover(self, event):
        """Handle mouse hover events to show point information and highlight categories"""
        if event.inaxes != self.ax or not self.visible_data_points:
            if self.hover_annotation:
                self.hover_annotation.set_visible(False)
            # Reset highlighting when mouse leaves the plot area
            self.reset_category_highlighting()
            return
        
        # Find the closest point to the mouse cursor
        if event.xdata is None or event.ydata is None:
            return
        
        closest_point = None
        min_distance = float('inf')
        hover_threshold = 0.02  # Adjust this to change hover sensitivity
        
        # Calculate the threshold in data coordinates
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        threshold_x = x_range * hover_threshold
        threshold_y = y_range * hover_threshold
        
        for point in self.visible_data_points:
            # Calculate distance from mouse to point
            dx = abs(event.xdata - point['x'])
            dy = abs(event.ydata - point['y'])
            
            # Use a simple distance metric
            distance = np.sqrt(dx**2 + dy**2)
            
            # Check if mouse is close enough to the point
            if dx < threshold_x and dy < threshold_y and distance < min_distance:
                min_distance = distance
                closest_point = point
        
        if closest_point:
            # Highlight the category of the closest point
            self.highlight_category(closest_point['category'])
            
            # Show annotation with category and object name
            self.hover_annotation.xy = (closest_point['x'], closest_point['y'])
            
            # Clean up object name (remove file extension if present)
            object_name = closest_point['object_name']
            if object_name.endswith('.obj'):
                object_name = object_name[:-4]
            
            text = f"Category: {closest_point['category']}\nObject: {object_name}"
            self.hover_annotation.set_text(text)
            self.hover_annotation.set_visible(True)
        else:
            # Reset highlighting and hide annotation if no point is close enough
            self.reset_category_highlighting()
            self.hover_annotation.set_visible(False)

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = TSNEVisualizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
