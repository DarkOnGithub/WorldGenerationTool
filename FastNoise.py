import ctypes
import sys
from typing import Dict, List, Tuple, Union
import numpy as np
class OutputMinMax:
    def __init__(self, min_value: float = float('inf'), max_value: float = float('-inf')):
        self.min = min_value
        self.max = max_value

    def __init__(self, native_output_min_max: List[float]):
        self.min = native_output_min_max[0]
        self.max = native_output_min_max[1]

    def merge(self, other: 'OutputMinMax'):
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
import time
class FastNoise:
    class Metadata:
        class Member:
            class Type:
                Float = 0
                Int = 1
                Enum = 2
                NodeLookup = 3
                Hybrid = 4

            def __init__(self):
                self.name = ""
                self.type = 0
                self.index = 0
                self.enum_names: Dict[str, int] = {}

        def __init__(self):
            self.id = 0
            self.name = ""
            self.members: Dict[str, 'FastNoise.Metadata.Member'] = {}

    def __init__(self, metadata_name: str | int):
        if isinstance(metadata_name, str):
            formatted_name = self._format_lookup(metadata_name)
            if formatted_name not in self.metadata_name_lookup:
                raise ValueError(f"Failed to find metadata name: {metadata_name}")
            self.metadata_id = self.metadata_name_lookup[formatted_name]
            # Pass both arguments: metadata_id and simdLevel (default to 0)
            self.node_handle = self._fn_new_from_metadata(self.metadata_id, 0)
            if not self.node_handle:
                raise RuntimeError("Failed to create FastNoise node from metadata")
        elif isinstance(metadata_name, int):
            self.metadata_id = self._fn_get_metadata_id(metadata_name)
            # Pass both arguments: metadata_id and simdLevel (default to 0)
            self.node_handle = metadata_name
            if not self.node_handle:
                raise RuntimeError("Failed to create FastNoise node from metadata")
    def __del__(self):
        # Check if node_handle exists before attempting to delete
        if hasattr(self, 'node_handle') and self.node_handle:
            self._fn_delete_node_ref(self.node_handle)
    @staticmethod
    def from_encoded_node_tree(encoded_node_tree: str) -> 'FastNoise':
        node_handle = FastNoise._fn_new_from_encoded_node_tree(ctypes.c_char_p(encoded_node_tree.encode('utf-8')), ctypes.c_uint(0))
        if not node_handle:
            return None
        return FastNoise(node_handle)

    def get_simd_level(self) -> int:
        return self._fn_get_simd_level(self.node_handle)

    def set(self, member_name: str, value: float):
        formatted_name = self._format_lookup(member_name)
        if formatted_name not in self.node_metadata[self.metadata_id].members:
            raise ValueError(f"Failed to find member name: {member_name}")
        member = self.node_metadata[self.metadata_id].members[formatted_name]
        if member.type == FastNoise.Metadata.Member.Type.Float:
            if not self._fn_set_variable_float(self.node_handle, member.index, value):
                raise RuntimeError("Failed to set float value")
        elif member.type == FastNoise.Metadata.Member.Type.Hybrid:
            if not self._fn_set_hybrid_float(self.node_handle, member.index, value):
                raise RuntimeError("Failed to set float value")
        else:
            raise ValueError(f"{member_name} cannot be set to a float value")

    def set_int(self, member_name: str, value: int):
        formatted_name = self._format_lookup(member_name)
        if formatted_name not in self.node_metadata[self.metadata_id].members:
            raise ValueError(f"Failed to find member name: {member_name}")
        member = self.node_metadata[self.metadata_id].members[formatted_name]
        if member.type != FastNoise.Metadata.Member.Type.Int:
            raise ValueError(f"{member_name} cannot be set to an int value")
        if not self._fn_set_variable_int_enum(self.node_handle, member.index, value):
            raise RuntimeError("Failed to set int value")

    def set_enum(self, member_name: str, enum_value: str):
        formatted_name = self._format_lookup(member_name)
        if formatted_name not in self.node_metadata[self.metadata_id].members:
            raise ValueError(f"Failed to find member name: {member_name}")
        member = self.node_metadata[self.metadata_id].members[formatted_name]
        if member.type != FastNoise.Metadata.Member.Type.Enum:
            raise ValueError(f"{member_name} cannot be set to an enum value")
        if enum_value not in member.enum_names:
            raise ValueError(f"Failed to find enum value: {enum_value}")
        if not self._fn_set_variable_int_enum(self.node_handle, member.index, member.enum_names[enum_value]):
            raise RuntimeError("Failed to set enum value")

    def set_node_lookup(self, member_name: str, node_lookup: 'FastNoise'):
        formatted_name = self._format_lookup(member_name)
        if formatted_name not in self.node_metadata[self.metadata_id].members:
            raise ValueError(f"Failed to find member name: {member_name}")
        member = self.node_metadata[self.metadata_id].members[formatted_name]
        if member.type == FastNoise.Metadata.Member.Type.NodeLookup:
            if not self._fn_set_node_lookup(self.node_handle, member.index, node_lookup.node_handle):
                raise RuntimeError("Failed to set node lookup")
        elif member.type == FastNoise.Metadata.Member.Type.Hybrid:
            if not self._fn_set_hybrid_node_lookup(self.node_handle, member.index, node_lookup.node_handle):
                raise RuntimeError("Failed to set node lookup")
        else:
            raise ValueError(f"{member_name} cannot be set to a node lookup")


    def gen_uniform_grid_2d(self, noise_out: Union[np.ndarray, List[float]], x_start: int, y_start: int, 
                            x_size: int, y_size: int, frequency: float, seed: int) -> OutputMinMax:
        if not isinstance(noise_out, np.ndarray):
            noise_out = np.array(noise_out, dtype=np.float32)
        elif noise_out.dtype != np.float32:
            noise_out = noise_out.astype(np.float32)
        
        noise_out_array = (ctypes.c_float * len(noise_out)).from_buffer(noise_out)
        min_max = (ctypes.c_float * 2)(0.0, 0.0)
        
        self._fn_gen_uniform_grid_2d(
            self.node_handle,
            noise_out_array,
            x_start, y_start,
            x_size, y_size,
            frequency, seed,
            min_max
        )
        
        return OutputMinMax((min_max[0], min_max[1]))

    def gen_uniform_grid_3d(self, noise_out: List[float], x_start: int, y_start: int, z_start: int, x_size: int, y_size: int, z_size: int, frequency: float, seed: int) -> OutputMinMax:
        min_max = [0.0, 0.0]
        self._fn_gen_uniform_grid_3d(self.node_handle, noise_out, x_start, y_start, z_start, x_size, y_size, z_size, frequency, seed, min_max)
        return OutputMinMax(min_max)

    def gen_uniform_grid_4d(self, noise_out: List[float], x_start: int, y_start: int, z_start: int, w_start: int, x_size: int, y_size: int, z_size: int, w_size: int, frequency: float, seed: int) -> OutputMinMax:
        min_max = [0.0, 0.0]
        self._fn_gen_uniform_grid_4d(self.node_handle, noise_out, x_start, y_start, z_start, w_start, x_size, y_size, z_size, w_size, frequency, seed, min_max)
        return OutputMinMax(min_max)

    def gen_tileable_2d(self, noise_out: List[float], x_size: int, y_size: int, frequency: float, seed: int) -> OutputMinMax:
        min_max = [0.0, 0.0]
        self._fn_gen_tileable_2d(self.node_handle, noise_out, x_size, y_size, frequency, seed, min_max)
        return OutputMinMax(min_max)

    def gen_position_array_2d(self, noise_out: List[float], x_pos_array: List[float], y_pos_array: List[float], x_offset: float, y_offset: float, seed: int) -> OutputMinMax:
        min_max = [0.0, 0.0]
        self._fn_gen_position_array_2d(self.node_handle, noise_out, len(x_pos_array), x_pos_array, y_pos_array, x_offset, y_offset, seed, min_max)
        return OutputMinMax(min_max)

    def gen_position_array_3d(self, noise_out: List[float], x_pos_array: List[float], y_pos_array: List[float], z_pos_array: List[float], x_offset: float, y_offset: float, z_offset: float, seed: int) -> OutputMinMax:
        min_max = [0.0, 0.0]
        self._fn_gen_position_array_3d(self.node_handle, noise_out, len(x_pos_array), x_pos_array, y_pos_array, z_pos_array, x_offset, y_offset, z_offset, seed, min_max)
        return OutputMinMax(min_max)

    def gen_position_array_4d(self, noise_out: List[float], x_pos_array: List[float], y_pos_array: List[float], z_pos_array: List[float], w_pos_array: List[float], x_offset: float, y_offset: float, z_offset: float, w_offset: float, seed: int) -> OutputMinMax:
        min_max = [0.0, 0.0]
        self._fn_gen_position_array_4d(self.node_handle, noise_out, len(x_pos_array), x_pos_array, y_pos_array, z_pos_array, w_pos_array, x_offset, y_offset, z_offset, w_offset, seed, min_max)
        return OutputMinMax(min_max)

    def gen_single_2d(self, x: float, y: float, seed: int) -> float:
        return self._fn_gen_single_2d(self.node_handle, x, y, seed)

    def gen_single_3d(self, x: float, y: float, z: float, seed: int) -> float:
        return self._fn_gen_single_3d(self.node_handle, x, y, z, seed)

    def gen_single_4d(self, x: float, y: float, z: float, w: float, seed: int) -> float:
        return self._fn_gen_single_4d(self.node_handle, x, y, z, w, seed)

    @staticmethod
    def _format_lookup(s: str) -> str:
        return s.replace(" ", "").lower()

    @staticmethod
    def _format_dimension_member(name: str, dim_idx: int) -> str:
        if dim_idx >= 0:
            dim_suffix = ['x', 'y', 'z', 'w']
            name += dim_suffix[dim_idx]
        return name

    metadata_name_lookup: Dict[str, int] = {}
    node_metadata: List[Metadata] = []

    NATIVE_LIB = "./FastNoise.dll"

    _fn_new_from_metadata = ctypes.CDLL(NATIVE_LIB).fnNewFromMetadata
    _fn_new_from_metadata.argtypes = [ctypes.c_int, ctypes.c_uint]
    _fn_new_from_metadata.restype = ctypes.c_void_p

    _fn_new_from_encoded_node_tree = ctypes.CDLL(NATIVE_LIB).fnNewFromEncodedNodeTree
    _fn_new_from_encoded_node_tree.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    _fn_new_from_encoded_node_tree.restype = ctypes.c_void_p

    _fn_delete_node_ref = ctypes.CDLL(NATIVE_LIB).fnDeleteNodeRef
    _fn_delete_node_ref.argtypes = [ctypes.c_void_p]
    _fn_delete_node_ref.restype = None

    _fn_get_simd_level = ctypes.CDLL(NATIVE_LIB).fnGetSIMDLevel
    _fn_get_simd_level.argtypes = [ctypes.c_void_p]
    _fn_get_simd_level.restype = ctypes.c_uint

    _fn_get_metadata_id = ctypes.CDLL(NATIVE_LIB).fnGetMetadataID
    _fn_get_metadata_id.argtypes = [ctypes.c_void_p]
    _fn_get_metadata_id.restype = ctypes.c_int

    _fn_gen_uniform_grid_2d = ctypes.CDLL(NATIVE_LIB).fnGenUniformGrid2D
    _fn_gen_uniform_grid_2d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _fn_gen_uniform_grid_2d.restype = ctypes.c_uint

    _fn_gen_uniform_grid_3d = ctypes.CDLL(NATIVE_LIB).fnGenUniformGrid3D
    _fn_gen_uniform_grid_3d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _fn_gen_uniform_grid_3d.restype = ctypes.c_uint

    _fn_gen_uniform_grid_4d = ctypes.CDLL(NATIVE_LIB).fnGenUniformGrid4D
    _fn_gen_uniform_grid_4d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _fn_gen_uniform_grid_4d.restype = ctypes.c_uint

    _fn_gen_tileable_2d = ctypes.CDLL(NATIVE_LIB).fnGenTileable2D
    _fn_gen_tileable_2d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _fn_gen_tileable_2d.restype = None

    _fn_gen_position_array_2d = ctypes.CDLL(NATIVE_LIB).fnGenPositionArray2D
    _fn_gen_position_array_2d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _fn_gen_position_array_2d.restype = None

    _fn_gen_position_array_3d = ctypes.CDLL(NATIVE_LIB).fnGenPositionArray3D
    _fn_gen_position_array_3d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _fn_gen_position_array_3d.restype = None

    _fn_gen_position_array_4d = ctypes.CDLL(NATIVE_LIB).fnGenPositionArray4D
    _fn_gen_position_array_4d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    _fn_gen_position_array_4d.restype = None

    _fn_gen_single_2d = ctypes.CDLL(NATIVE_LIB).fnGenSingle2D
    _fn_gen_single_2d.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int]
    _fn_gen_single_2d.restype = ctypes.c_float

    _fn_gen_single_3d = ctypes.CDLL(NATIVE_LIB).fnGenSingle3D
    _fn_gen_single_3d.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]
    _fn_gen_single_3d.restype = ctypes.c_float

    _fn_gen_single_4d = ctypes.CDLL(NATIVE_LIB).fnGenSingle4D
    _fn_gen_single_4d.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]
    _fn_gen_single_4d.restype = ctypes.c_float

    _fn_get_metadata_count = ctypes.CDLL(NATIVE_LIB).fnGetMetadataCount
    _fn_get_metadata_count.argtypes = []
    _fn_get_metadata_count.restype = ctypes.c_int

    _fn_get_metadata_name = ctypes.CDLL(NATIVE_LIB).fnGetMetadataName
    _fn_get_metadata_name.argtypes = [ctypes.c_int]
    _fn_get_metadata_name.restype = ctypes.c_char_p

    _fn_get_metadata_variable_count = ctypes.CDLL(NATIVE_LIB).fnGetMetadataVariableCount
    _fn_get_metadata_variable_count.argtypes = [ctypes.c_int]
    _fn_get_metadata_variable_count.restype = ctypes.c_int

    _fn_get_metadata_variable_name = ctypes.CDLL(NATIVE_LIB).fnGetMetadataVariableName
    _fn_get_metadata_variable_name.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_variable_name.restype = ctypes.c_char_p

    _fn_get_metadata_variable_type = ctypes.CDLL(NATIVE_LIB).fnGetMetadataVariableType
    _fn_get_metadata_variable_type.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_variable_type.restype = ctypes.c_int

    _fn_get_metadata_variable_dimension_idx = ctypes.CDLL(NATIVE_LIB).fnGetMetadataVariableDimensionIdx
    _fn_get_metadata_variable_dimension_idx.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_variable_dimension_idx.restype = ctypes.c_int

    _fn_get_metadata_enum_count = ctypes.CDLL(NATIVE_LIB).fnGetMetadataEnumCount
    _fn_get_metadata_enum_count.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_enum_count.restype = ctypes.c_int

    _fn_get_metadata_enum_name = ctypes.CDLL(NATIVE_LIB).fnGetMetadataEnumName
    _fn_get_metadata_enum_name.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_enum_name.restype = ctypes.c_char_p

    _fn_set_variable_float = ctypes.CDLL(NATIVE_LIB).fnSetVariableFloat
    _fn_set_variable_float.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    _fn_set_variable_float.restype = ctypes.c_bool

    _fn_set_variable_int_enum = ctypes.CDLL(NATIVE_LIB).fnSetVariableIntEnum
    _fn_set_variable_int_enum.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _fn_set_variable_int_enum.restype = ctypes.c_bool

    _fn_get_metadata_node_lookup_count = ctypes.CDLL(NATIVE_LIB).fnGetMetadataNodeLookupCount
    _fn_get_metadata_node_lookup_count.argtypes = [ctypes.c_int]
    _fn_get_metadata_node_lookup_count.restype = ctypes.c_int

    _fn_get_metadata_node_lookup_name = ctypes.CDLL(NATIVE_LIB).fnGetMetadataNodeLookupName
    _fn_get_metadata_node_lookup_name.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_node_lookup_name.restype = ctypes.c_char_p

    _fn_get_metadata_node_lookup_dimension_idx = ctypes.CDLL(NATIVE_LIB).fnGetMetadataNodeLookupDimensionIdx
    _fn_get_metadata_node_lookup_dimension_idx.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_node_lookup_dimension_idx.restype = ctypes.c_int

    _fn_set_node_lookup = ctypes.CDLL(NATIVE_LIB).fnSetNodeLookup
    _fn_set_node_lookup.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    _fn_set_node_lookup.restype = ctypes.c_bool

    _fn_get_metadata_hybrid_count = ctypes.CDLL(NATIVE_LIB).fnGetMetadataHybridCount
    _fn_get_metadata_hybrid_count.argtypes = [ctypes.c_int]
    _fn_get_metadata_hybrid_count.restype = ctypes.c_int

    _fn_get_metadata_hybrid_name = ctypes.CDLL(NATIVE_LIB).fnGetMetadataHybridName
    _fn_get_metadata_hybrid_name.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_hybrid_name.restype = ctypes.c_char_p

    _fn_get_metadata_hybrid_dimension_idx = ctypes.CDLL(NATIVE_LIB).fnGetMetadataHybridDimensionIdx
    _fn_get_metadata_hybrid_dimension_idx.argtypes = [ctypes.c_int, ctypes.c_int]
    _fn_get_metadata_hybrid_dimension_idx.restype = ctypes.c_int

    _fn_set_hybrid_node_lookup = ctypes.CDLL(NATIVE_LIB).fnSetHybridNodeLookup
    _fn_set_hybrid_node_lookup.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    _fn_set_hybrid_node_lookup.restype = ctypes.c_bool

    _fn_set_hybrid_float = ctypes.CDLL(NATIVE_LIB).fnSetHybridFloat
    _fn_set_hybrid_float.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    _fn_set_hybrid_float.restype = ctypes.c_bool

    @staticmethod
    def _initialize_metadata():
        metadata_count = FastNoise._fn_get_metadata_count()
        FastNoise.node_metadata = [FastNoise.Metadata() for _ in range(metadata_count)]
        FastNoise.metadata_name_lookup = {}

        for id in range(metadata_count):
            metadata = FastNoise.Metadata()
            metadata.id = id
            metadata.name = FastNoise._format_lookup(FastNoise._fn_get_metadata_name(id).decode('utf-8'))
            FastNoise.metadata_name_lookup[metadata.name] = id

            variable_count = FastNoise._fn_get_metadata_variable_count(id)
            node_lookup_count = FastNoise._fn_get_metadata_node_lookup_count(id)
            hybrid_count = FastNoise._fn_get_metadata_hybrid_count(id)
            metadata.members = {}

            for variable_idx in range(variable_count):
                member = FastNoise.Metadata.Member()
                member.name = FastNoise._format_lookup(FastNoise._fn_get_metadata_variable_name(id, variable_idx).decode('utf-8'))
                member.type = FastNoise._fn_get_metadata_variable_type(id, variable_idx)
                member.index = variable_idx
                member.name = FastNoise._format_dimension_member(member.name, FastNoise._fn_get_metadata_variable_dimension_idx(id, variable_idx))

                if member.type == FastNoise.Metadata.Member.Type.Enum:
                    enum_count = FastNoise._fn_get_metadata_enum_count(id, variable_idx)
                    member.enum_names = {}
                    for enum_idx in range(enum_count):
                        enum_name = FastNoise._format_lookup(FastNoise._fn_get_metadata_enum_name(id, variable_idx, enum_idx).decode('utf-8'))
                        member.enum_names[enum_name] = enum_idx

                metadata.members[member.name] = member

            for node_lookup_idx in range(node_lookup_count):
                member = FastNoise.Metadata.Member()
                member.name = FastNoise._format_lookup(FastNoise._fn_get_metadata_node_lookup_name(id, node_lookup_idx).decode('utf-8'))
                member.type = FastNoise.Metadata.Member.Type.NodeLookup
                member.index = node_lookup_idx
                member.name = FastNoise._format_dimension_member(member.name, FastNoise._fn_get_metadata_node_lookup_dimension_idx(id, node_lookup_idx))
                metadata.members[member.name] = member

            for hybrid_idx in range(hybrid_count):
                member = FastNoise.Metadata.Member()
                member.name = FastNoise._format_lookup(FastNoise._fn_get_metadata_hybrid_name(id, hybrid_idx).decode('utf-8'))
                member.type = FastNoise.Metadata.Member.Type.Hybrid
                member.index = hybrid_idx
                member.name = FastNoise._format_dimension_member(member.name, FastNoise._fn_get_metadata_hybrid_dimension_idx(id, hybrid_idx))
                metadata.members[member.name] = member

            FastNoise.node_metadata[id] = metadata

FastNoise._initialize_metadata()
