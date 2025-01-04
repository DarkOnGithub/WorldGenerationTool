import os
from abc import ABC, abstractmethod
import dearpygui.dearpygui as dpg
import numpy as np
from FastNoise import FastNoise
from typing import Tuple
from dataclasses import dataclass
from Spline import Spline



def clamp(n, min, max): 
    if n < min: 
        return min
    elif n > max: 
        return max
    else: 
        return n 
    
@dataclass
class ViewState:
    position: Tuple[float, float] = (0.0, 0.0)
    is_dragging: bool = False
    last_mouse_pos: Tuple[float, float] = (0.0, 0.0)
    sensitivity: float = 1

class WorldGenerationComponent(ABC):
    
    def __init__(self, path: str = ""):
        self.path = path
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def on_item_selected(self):
        pass
    
    @classmethod
    def render_selection(cls, files: list, parent: int, component_class: type):

        for file in files:
            component = component_class(file)
            dpg.add_selectable(
                label=component.get_name(),
                parent=parent,
                callback=component.on_item_selected
            )

class NoiseComponent(WorldGenerationComponent):
    
    RELATIVE_PATH = "Noise/"
    
    def get_name(self) -> str:
        return os.path.basename(self.path)
    
    def on_item_selected(self):
        print(f"Selected noise component: {self.get_name()}")
    
    @classmethod
    def get_files(cls, directory: str) -> list:

        search_dir = os.path.join(directory, cls.RELATIVE_PATH)
        if not os.path.exists(search_dir):
            return []
            
        files = []
        for root, _, filenames in os.walk(search_dir):
            files.extend(
                os.path.join(root, filename)
                for filename in filenames
                if os.path.isfile(os.path.join(root, filename))
            )
        return files

class PreviewSubWindow(ABC):
    SIZE = (-1, 512)
    window = None
    
    def __init__(self, title: str):
        self.title = title
        self.tab = None
    
    @abstractmethod
    def get_tab_name(self) -> str:
        return self.title
    
    @abstractmethod
    def render(self, parent: int):
        pass


    def _get_window_width(self, window) -> int:
        w = dpg.get_item_width(window)
        if w == -1 or w == 0:
            return self._get_window_width(dpg.get_item_parent(window))
        return w
    
    def _get_window_height(self, window) -> int:
        h = dpg.get_item_height(window)
        print(dpg.get_item_alias(window))
        
        if h == -1 or h == 0:
            return self._get_window_height(dpg.get_item_parent(window))
        return h
    
    def get_window_size(self) -> Tuple[int, int]:
        return (self._get_window_width(self.window), self._get_window_height(self.window))

    def get_tag(self, tag: str) -> str:
        return f"{tag}{id(self)}"
    
    
    def show(self):
        dpg.show_item(self.window)
        
    def hide(self):
        dpg.hide_item(self.window)
        
class NoisePreview(PreviewSubWindow):
    SIZE = (512, 512)
    window_size = None
    def __init__(self, noise_name: str):
        self.noise_name = noise_name
        self.current_noise = None
        self.view_state = ViewState()
        self.buffer = np.zeros(self.SIZE[0] * self.SIZE[1], dtype=np.float32)
        self.spline = None
    def _handle_drag(self, sender: int, mouse_data: Tuple[int, float, float]) -> None:
        if not dpg.is_item_hovered(self.window) or not all(mouse_data[1:]):
            return
        button, x, y = mouse_data
        if button != 0:
            return

        if not self.view_state.is_dragging:
            self.view_state.is_dragging = True
            self.view_state.last_mouse_pos = (x, y)
        else:
            delta_x = self.view_state.last_mouse_pos[0] - x
            delta_y = self.view_state.last_mouse_pos[1] - y
            
            self.view_state.position = (
                self.view_state.position[0] + delta_x * self.view_state.sensitivity,
                self.view_state.position[1] + delta_y * self.view_state.sensitivity
            )
            self.view_state.last_mouse_pos = (x, y)
            self.update_noise(self.current_noise)

    def _handle_release(self, sender: int, mouse_data: Tuple[int, float, float]) -> None:
        self.view_state.is_dragging = False



    def render(self, parent: int):
        with dpg.child_window(label=self.noise_name, width=self.SIZE[0], height=self.SIZE[1] + 75, parent=parent) as self.window:
            
            dpg.add_input_float(label="Frequency", default_value=0.001, width=self.SIZE[0] - 120, tag=self.get_tag("frequency"), step=0.001, min_value=0, callback=lambda: self.update_noise(self.current_noise))
            dpg.add_input_int(label="Seed", default_value=42, tag=self.get_tag("seed"), callback=lambda: self.update_noise(self.current_noise))
            
            with dpg.texture_registry(show=False): 
                dpg.add_raw_texture(
                    width=self.SIZE[0], 
                    height=self.SIZE[1], 
                    tag=self.get_tag("noise_texture"), 
                    format=dpg.mvFormat_Float_rgb, 
                    default_value=np.repeat(self.buffer, 3),
                )
        
            dpg.add_image(self.get_tag("noise_texture"))            
            with dpg.handler_registry():
                dpg.add_mouse_drag_handler(callback=self._handle_drag)
                dpg.add_mouse_release_handler(callback=self._handle_release)
                
            dpg.bind_item_handler_registry(self.window, "handler_registry")
        self.hide()
        
    def update_noise(self, noise: FastNoise) -> None:
        dpg.set_value(self.get_tag("frequency"), clamp(dpg.get_value(self.get_tag("frequency")), 0, 2**32))
        self.current_noise = noise
        
        noise.gen_uniform_grid_2d(
            self.buffer,
            int(self.view_state.position[0]),
            int(self.view_state.position[1]),
            self.SIZE[0],
            self.SIZE[1],
            dpg.get_value(self.get_tag("frequency")),
            dpg.get_value(self.get_tag("seed"))
        )
        if self.spline:
            self.buffer = self.spline.evaluate(self.buffer)
        dpg.set_value(self.get_tag("noise_texture"), np.repeat(self.buffer, 3))
    def get_tab_name(self) -> str:
        return self.noise_name


class PreviewWindow:
    windows = {}
    tab_bar = None
    window = None
    current_tab = None

    @classmethod
    def render(cls):
        with dpg.window(label="Preview", tag="preview_window") as cls.window:
            cls.tab_bar = dpg.add_tab_bar(
                tag="preview_tab_bar",
                show=False,
                reorderable=True
            )
            with dpg.child_window(tag="Container", always_auto_resize=True) as cls.container:
                with dpg.group() as cls.group_container:
                    pass
            with dpg.item_handler_registry(tag="tab_bar_handler"):
                dpg.add_item_clicked_handler(callback=cls._on_click)

    @classmethod
    def _on_click(cls, sender: int, app_data: int):
        if app_data[0] == 1:
            cls._open_popup(app_data[1])
        elif app_data[0] == 0:
            cls._open_tab(app_data[1])

    @classmethod
    def _open_popup(cls, sender: int):
        with dpg.popup(sender, mousebutton=dpg.mvMouseButton_Right, modal=False):
            dpg.add_menu_item(label="Move into this tab", callback=lambda: cls._extend_tab(sender))

    @classmethod
    def _extend_tab(cls, tab_group: int):
        if cls.current_tab is None or tab_group not in cls.windows:
            return
        if tab_group in cls.windows:
            cls.windows[cls.current_tab].extend(cls.windows[tab_group])
            dpg.delete_item(tab_group)
            del cls.windows[tab_group]
            cls._open_tab(cls.current_tab)

    @classmethod
    def _open_tab(cls, tab: int):
        if cls.current_tab:
            for window in cls.windows[cls.current_tab]:
                dpg.hide_item(window.window)
        cls.current_tab = tab
        for window in cls.windows.get(tab, []):
            dpg.show_item(window.window)

    @classmethod
    def add_window(cls, sub_window: PreviewSubWindow):
        with dpg.tab(label=sub_window.get_tab_name(), parent=cls.tab_bar) as tab:
            sub_window.tab = tab
            cls.windows[tab] = [sub_window]
            dpg.bind_item_handler_registry(tab, "tab_bar_handler")
            
        sub_window.render(cls.group_container)
        if cls.current_tab is None:
                cls._open_tab(tab)
        cls.update_tab_bar()

    @classmethod
    def remove_window(cls, sub_window: PreviewSubWindow):
        for tab, windows in cls.windows.items():
            if sub_window in windows:
                windows.remove(sub_window)
                if not windows:
                    dpg.delete_item(tab)
                    del cls.windows[tab]
        dpg.delete_item(sub_window.window)
        cls.update_tab_bar()

    @classmethod
    def update_tab_bar(cls):
        if cls.windows:
            dpg.show_item(cls.tab_bar)
        else:
            dpg.hide_item(cls.tab_bar)

class SplinePreview(PreviewSubWindow):
    SIZE = (512, 512)
    def __init__(self, noise_preview: NoisePreview):
        super().__init__(f"spline_{noise_preview.get_tab_name()}")
        self.noise_preview = noise_preview
        self.points = {}
        self.point_id = 0
        self.line = None
        self.plot = None
        self.mode = 0
        spline = Spline([0, 0, 1, 1])
        self.spline = spline
        noise_preview.spline =  spline

    def clamp(self, value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(value, max_val))

    def on_point_move(self, sender, data):
        if self.mode == 1:
            self.spline.remove_point(dpg.get_value(sender)[0])
            del self.points[dpg.get_item_alias(sender)]
            dpg.delete_item(sender)
            self.update_lines()
            self.mode = 0
            return

        x_min_max = dpg.get_axis_limits("x_axis")
        y_min_max = dpg.get_axis_limits("y_axis")

        new_pos = [
            self.clamp(dpg.get_value(sender)[0], x_min_max[0], x_min_max[1]),
            self.clamp(dpg.get_value(sender)[1], y_min_max[0], y_min_max[1])
        ]

        dpg.set_value(sender, new_pos)
        item_name = dpg.get_item_alias(sender)

        if self.points[item_name] != new_pos:
            self.spline.remove_control_point(self.points[item_name][0])

            self.points[item_name] = new_pos
            self.spline.add_control_points([new_pos])
            self.update_lines()
            self.noise_preview.update_noise(self.noise_preview.current_noise)

    def add_point(self, x: float, y: float):
        point_tag = f"point_{self.point_id}"

        point_id = dpg.add_drag_point(
            color=[255, 255, 255, 255],
            default_value=[x, y],
            callback=self.on_point_move,
            parent=self.plot,
            tag=point_tag
        )

        self.point_id += 1
        self.points[point_id] = [x, y]
        self.spline.add_control_points([(x, y)])
        self.update_lines()

    def remove_point(self, x: float, y: float):
        self.mode = 1

    def update_lines(self):
        if self.line:
            dpg.delete_item(self.line)
            self.line = None

        points = list(self.points.values())
        if len(points) < 2:
            return

        sorted_points = sorted(points, key=lambda point: point[0])
        lines_x = [point[0] for point in sorted_points]
        lines_y = [point[1] for point in sorted_points]

        self.line = dpg.add_line_series(
            x=lines_x,
            y=lines_y,
            label="",
            parent="x_axis"
        )

    def render(self, parent: int):
        with dpg.child_window(label=self.title, width=self.SIZE[0], height=self.SIZE[1] + 75, parent=parent) as self.window:
            with dpg.plot(height=self.SIZE[0], width=self.SIZE[1]) as plot:
                self.plot = plot
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="x_axis")
                dpg.set_axis_limits(x_axis, 0, 1)
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="y_axis")
                dpg.set_axis_limits(y_axis, 0, 1)
                self.add_point(0, 0)
                self.add_point(1, 1)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Add Point", callback=lambda: self.add_point(0.5, 0.5))
                dpg.add_button(label="Remove Point", callback=lambda: self.remove_point(0, 0))
        self.hide()
        
    def show(self):
        dpg.show_item(self.window)

    def hide(self):
        dpg.hide_item(self.window)
        
    def get_tab_name(self):
        return f"spline_{self.noise_preview.get_tab_name()}"
class ContentSelector:
    
    window = None
    button_theme = None
    
    @classmethod
    def render(cls):
        with dpg.window(label="Content Selector") as cls.window:
            cls._setup_theme()
            cls._setup_file_dialog()
            cls._create_menu_bar()
    
    @classmethod
    def _setup_theme(cls):
        with dpg.theme() as cls.button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
    
    @classmethod
    def _setup_file_dialog(cls):
        dpg.add_file_dialog(
            directory_selector=True,
            show=False,
            callback=cls._on_directory_selected,
            tag="directory_selector",
            width=700,
            height=400
        )
    
    @classmethod
    def _create_menu_bar(cls):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Files"):
                dpg.add_menu_item(label="Open", callback=cls._open_directory)
                dpg.add_menu_item(label="Save")
                dpg.add_menu_item(label="Save As")
                dpg.add_menu_item(label="Load Preset")
    
    @classmethod
    def _on_directory_selected(cls, sender: int, app_data: dict):
        with dpg.tree_node(label="Noises", parent=cls.window) as tree:
            files = NoiseComponent.get_files(app_data["file_path_name"])
            NoiseComponent.render_selection(files, tree, NoiseComponent)
    
    @classmethod
    def _open_directory(cls):
        dpg.show_item("directory_selector")

class WorldGenerationTool:
    
    def __init__(self):
        self._setup_context()
        self._configure_app()
        self._create_windows()
        self._run_app()
    
    def _setup_context(self):
        dpg.create_context()
        dpg.create_viewport(
            width=1920,
            height=1080,
            title='World Generation Tool'
        )
        dpg.setup_dearpygui()
    
    def _configure_app(self):
        dpg.configure_app(
            docking=True,
            docking_space=True,
            init_file="config.ini",
            load_init_file=True
        )
    
    def _create_windows(self):
        ContentSelector.render()
        PreviewWindow.render()
        prev = NoisePreview("Erosion")
        PreviewWindow.add_window(prev)
        prev.update_noise(FastNoise.from_encoded_node_tree("FwAAAIC/AACAPwAAAAAAAIA/CQA="))
        prev = SplinePreview(prev)
        PreviewWindow.add_window(prev)
    def _run_app(self):
        dpg.set_exit_callback(self._on_exit)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def _on_exit(self):
        dpg.save_init_file("config.ini")

# Initialize the spline with control points
control_points = [0.0, 1.0, 1.0, 0.0]  # Control points (x, y)
spline = Spline(control_points)

# Buffer is a NumPy array of noise values between 0 and 1
buffer = np.random.rand(1000).astype(np.float32)

# Evaluate the spline at all buffer points
buffer_processed = spline.evaluate(buffer)

print("Processed buffer:", all(buffer_processed == buffer))
if __name__ == "__main__":
    WorldGenerationTool()