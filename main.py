import os
from abc import ABC, abstractmethod
import dearpygui.dearpygui as dpg
import numpy as np
from FastNoise import FastNoise
from typing import Tuple
from dataclasses import dataclass




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
        print(dpg.get_item_alias(window))
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
    
class NoisePreview(PreviewSubWindow):
    SIZE = (512, 512)
    window_size = None
    def __init__(self, noise_name: str):
        self.noise_name = noise_name
        self.current_noise = None
        self.view_state = ViewState()
        self.buffer = np.zeros(self.SIZE[0] * self.SIZE[1], dtype=np.float32)
        
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
        
        dpg.set_value(self.get_tag("noise_texture"), np.repeat(self.buffer, 3))

    def get_tab_name(self) -> str:
        return self.noise_name


@dataclass
class PreviewTab:
    tab: int
    sub_window: List[PreviewSubWindow]

class PreviewWindow:
    
    windows = []
    tab_bar = None
    window = None
    
    @classmethod
    def render(cls):
        with dpg.window(label="Preview", tag="preview_window") as cls.window:
            cls.tab_bar = dpg.add_tab_bar(
                tag="preview_tab_bar",
                show=False,
                reorderable=True,
                callback=cls._open_popup
            )
            with dpg.child_window(tag="Container", always_auto_resize=True) as cls.container:
                with dpg.group() as cls.group_container:
                    pass
    
    @classmethod
    def _open_popup(cls, sender: int, app_data: any):
        with dpg.popup(app_data):
            dpg.add_menu_item(label="Add to group")
    
    @classmethod
    def add_window(cls, sub_window: PreviewSubWindow):
        with dpg.tab(label=sub_window.get_tab_name(), parent=cls.tab_bar) as tab:
            sub_window.tab = tab

        cls.windows.append(PreviewTab(tab, sub_window))
        cls.update_tab_bar()
    
    @classmethod
    def remove_window(cls, sub_window: PreviewSubWindow):
        if sub_window in cls.windows:
            for tab in cls.windows:
                if tab.sub_window == sub_window:
                    cls.windows.remove(tab)
            dpg.delete_item(sub_window.tab)
            cls.update_tab_bar()
    
    @classmethod
    def update_tab_bar(cls):
        if cls.windows:
            dpg.show_item(cls.tab_bar)
        else:
            dpg.hide_item(cls.tab_bar)

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
        prev = NoisePreview("Erosion")
        PreviewWindow.add_window(prev)
        prev.update_noise(FastNoise.from_encoded_node_tree("FwAAAIC/AACAPwAAAAAAAIA/CQA="))
        prev = NoisePreview("Erosion")
        PreviewWindow.add_window(prev)
        prev.update_noise(FastNoise.from_encoded_node_tree("FwAAAIC/AACAPwAAAAAAAIA/CQA="))
        prev = NoisePreview("Erosion")
        PreviewWindow.add_window(prev)
        prev.update_noise(FastNoise.from_encoded_node_tree("FwAAAIC/AACAPwAAAAAAAIA/CQA="))
    def _run_app(self):
        dpg.set_exit_callback(self._on_exit)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def _on_exit(self):
        dpg.save_init_file("config.ini")

if __name__ == "__main__":
    WorldGenerationTool()