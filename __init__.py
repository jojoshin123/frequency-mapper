bl_info = {
    "name" : "Frequency Mapper",
    "author" : "Jonah Shin",
    "description" : "Map audio frequency to object keyframes",
    "blender" : (4, 4, 0),
    "version" : (1, 1, 1),
    "location" : "View3D > Frequency Mapper",
    "warning" : "",
    "category" : "Animation"
}

import bpy
from .frequency_mapper import *

classes = (
    # TODO: add frequency_mapper.py classes
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # TODO: might have to add Scene props here

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)