import bpy

# ------------------------------------------------------------------------
#   Operator (process the chosen file)
# ------------------------------------------------------------------------

# Properties (holds file path)
class FM_Properties(bpy.types.PropertyGroup):
    audio_file: bpy.props.StringProperty(
        name="Audio File",
        description="Select an audio file",
        default="",
        subtype='FILE_PATH'
    )


class FM_OT_ProcessAudio(bpy.types.Operator):
    bl_idname = "fm.process_audio"
    bl_label = "Process Audio"
    bl_description = "Split audio into frequency ranges and animate objects"

    def execute(self, context):
        props = context.scene.fm_props
        filepath = props.audio_file
        filename = filepath.split("\\")[-1]  # Just the filename

        if not filepath:
            self.report({'ERROR'}, "No audio file selected!")
            return {'CANCELLED'}

        # --- Add to Video Sequencer
        try:
            # Ensure sequencer exists
            if not context.scene.sequence_editor:
                context.scene.sequence_editor_create()

            sequencer = context.scene.sequence_editor
            sequencer.sequences.new_sound(
                name=filename.split(".")[0],  # Name without extension
                filepath=filepath,
                channel=1,
                frame_start=1
            )

            self.report({'INFO'}, f"Imported {filepath} into VSE")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to import into VSE: {e}")
            return {'CANCELLED'}

        # Placeholder for your librosa/numpy processing
        self.report({'INFO'}, f"Processing audio: {filepath}")
        print(f"[DEBUG] Audio processing would happen here: {filepath}")

        return {'FINISHED'}

# ------------------------------------------------------------------------
#   Main Panel (UI)
# ------------------------------------------------------------------------
class FM_PT_MainPanel(bpy.types.Panel):
    bl_label = "Frequency Mapper"
    bl_idname = "FM_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FreqMap'

    def draw(self, context):
        layout = self.layout
        props = context.scene.fm_props

        layout.label(text="Audio → Animation Mapper")

        # File picker
        layout.prop(props, "audio_file")

        # Process button
        layout.operator("fm.process_audio", text="Process Audio")






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

# import bpy
# from .virtucamera_blender import *

classes = (
    FM_PT_MainPanel,
    FM_OT_ProcessAudio,
    FM_Properties
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.fm_props = bpy.props.PointerProperty(type=FM_Properties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.fm_props

if __name__ == "__main__":
    register()