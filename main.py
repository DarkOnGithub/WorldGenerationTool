import dearpygui.dearpygui as dpg
from abc import ABC, abstractmethod
import os

    
class WorldGenerationComponent:
    RELATIVE_PATH = ""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_name(self):
        pass
    
    @abstractmethod
    def on_item_selected(self):
        pass
    
    
    @classmethod
    def get_files():
        pass
    
    
    @classmethod
    def render_selection(self, files, parent, class_type):
        for file in files:
            component = class_type(file)
            dpg.add_selectable(label=component.get_name(), parent=parent, callback=component.on_item_selected)



class NoiseComponent(WorldGenerationComponent):
    RELATIVE_PATH = "Noise/"
    
    def __init__(self, path):
        super().__init__()
        self.path = path
    
    def get_name(self):
        return os.path.basename(self.path)
    
    def on_item_selected(self):
    
        print(self.get_name())
    @classmethod
    def get_files(self, directory):    
        directory = os.path.join(directory, self.RELATIVE_PATH)
        files = []
        
        if not os.path.exists(directory):
            return files
            
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                if os.path.isfile(full_path):
                    files.append(full_path)
                    
        return files
class PreviewWindow:
    windows = []
    tab_bar = None
    window = None
    
    @classmethod
    def render(self):
        with dpg.window(label="Preview", tag="preview_window") as self.window:
            self.tab_bar = dpg.add_tab_bar(tag="preview_tab_bar", show=False, reorderable=True)
    
    @classmethod
    def add_window(self, sub_window):
        with dpg.tab(label=sub_window.title, parent=self.tab_bar) as tab:
            sub_window.tab = tab
        self.windows.append(sub_window)
        self.update_tab_bar()
    
    @classmethod
    def remove_window(self, sub_window):
        if sub_window in self.windows:
            self.windows.remove(sub_window)
            dpg.delete_item(sub_window.tab)
            self.update_tab_bar()
    
    @classmethod 
    def update_tab_bar(self):
        if len(self.windows) == 0:
            dpg.hide_item(self.tab_bar)
        else:
            dpg.show_item(self.tab_bar)
            
class PreviewSubWindow:
    def __init__(self, title, size):
        self.title = title
        self.size = size
        
class ContentSelector:        
            
    @classmethod
    def _on_directory_selected(self, sender, app_data):
        with dpg.tree_node(label="Noises", parent=self.window) as tree:
            files = NoiseComponent.get_files(app_data["file_path_name"])
            NoiseComponent.render_selection(files, tree, NoiseComponent)

    @classmethod
    def _open_directory(self):
        dpg.show_item("directory_selector")
        
    @classmethod
    def render(self):
        with dpg.window(label="Content Selector") as self.window:
            with dpg.theme() as self.button_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0]) 
            dpg.add_file_dialog(
                directory_selector=True, show=False, callback=self._on_directory_selected, 
                tag="directory_selector", width=700 ,height=400
            )
            
            with dpg.viewport_menu_bar():
                with dpg.menu(label="Files"):
                    dpg.add_menu_item(label="Open", callback=self._open_directory)
                    dpg.add_menu_item(label="Save")
                    dpg.add_menu_item(label="Save As")
                    dpg.add_menu_item(label="Load Preset")
        
        


class WorldGenerationTool:
    def __init__(self):
        dpg.create_context()

        dpg.configure_app(docking=True, docking_space=True)
        dpg.configure_app(init_file="config.ini", load_init_file=True)
        dpg.create_viewport(width=1920, height=1080, title='World Generation Tool')
        dpg.setup_dearpygui()

        ContentSelector.render()
        PreviewWindow.render()
        p = PreviewSubWindow("Noise", (800, 600))
        PreviewWindow.add_window(p)
        p = PreviewSubWindow("Noise2", (800, 600))
        PreviewWindow.add_window(p)
        
        dpg.set_exit_callback(self._on_exit)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _on_exit(self):
        dpg.save_init_file("config.ini")
WorldGenerationTool()