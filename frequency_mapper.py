import bpy
import numpy as np
import librosa
# import pip
# pip.main(['install', 'librosa', '--user'])

import sys

packages_path = "\\Users\\jonahshin\\.local\\ bin" + "\\..\\site-packages"
sys.path.insert(0, packages_path)


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

    # Add this new text input property
    object_count: bpy.props.IntProperty(
        name="Object Count",
        description="Number of objects to map",
        default=0
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

        # Librosa processing
        self.report({'INFO'}, f"Processing audio: {filepath}")
        number_value = props.object_count

        numpy_data = mp3_to_numpy_librosa(filepath)
        arr = analyze_frequencies_sliding_window(numpy_data[0], numpy_data[1], n_bands=number_value, window_size=1024,
                                                 hop_size=512)

        fps = bpy.context.scene.render.fps
        frames = [int(t * fps) for t in arr[1]]

        arr_len = len(arr[0])


        objects = []
        for i in range(1,number_value+1):

            # TEMP!!!########################################################################
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            cube.name = str(i)
            cube.location = (i*2, 0.0, 0.0)
            #################################################################################

            objects.append(bpy.data.objects[str(i)])
            objects[-1].animation_data_clear()

        epsilon = 1e-6  # for log adjustment
        for i in range(arr_len):
            for j, obj in enumerate(objects):
                # Apply log scaling
                amp = arr[0][i][j]
                log_amp = np.log1p(amp + epsilon)  # log(1 + x)
                z_scale = log_amp * 2.0

                if j == 0:
                    print(f"\n\n\n{amp}\n\n\n")

                obj.scale.z = z_scale
                obj.keyframe_insert(data_path="scale", index=2, frame=frames[i])

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

        # Object count
        layout.prop(props, "object_count")

        # Process button
        layout.operator("fm.process_audio", text="Process Audio")




def analyze_frequencies_sliding_window(audio_data, sample_rate, n_bands=8, window_size=1024, hop_size=512):
    """
    Split audio into frequency bands over time using sliding window FFT

    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate
        n_bands: number of frequency bands
        window_size: size of each analysis window
        hop_size: how much to advance the window each step

    Returns:
        time_frequency_data: 2D array [time_frames, frequency_bands]
        time_points: array of time points for each frame
        band_ranges: frequency ranges for each band
    """

    # Set up frequency bands
    min_freq = 20
    max_freq = min(sample_rate // 2, 20000)

    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Determine how many bins per octave to cover the range
    n_octaves = int(np.ceil(np.log2(max_freq / min_freq)))
    bins_per_octave = n_bands // n_octaves if n_octaves > 0 else n_bands

    # Run CQT
    cqt_result = librosa.cqt(
        audio_data,
        sr=sample_rate,
        hop_length=hop_size,
        fmin=min_freq,
        n_bins=n_bands,
        bins_per_octave=12
    )
    magnitude = np.abs(cqt_result)

    # Transpose so shape = (time_frames, freq_bins)
    time_frequency_data = magnitude.T

    # Time points for each frame
    time_points = librosa.frames_to_time(np.arange(time_frequency_data.shape[0]),
                                         sr=sample_rate, hop_length=hop_size)

    # Band ranges: get actual CQT center freqs
    freqs = librosa.cqt_frequencies(n_bins=n_bands, fmin=min_freq, bins_per_octave=bins_per_octave)
    band_ranges = [(freqs[i], freqs[i + 1] if i + 1 < len(freqs) else freqs[i] * 2) for i in range(len(freqs))]

    return time_frequency_data, time_points, band_ranges


##########################################################################################################################
def mp3_to_numpy_librosa(filepath):
    """
    Load MP3 directly as numpy array using librosa
    Install: pip install librosa

    Returns:
        audio_array: numpy array of audio samples (float32, range -1 to 1)
        sample_rate: int, samples per second
    """

    try:
        import librosa

        # Load as numpy array (mono by default)
        audio_array, sample_rate = librosa.load(filepath, sr=None, dtype=np.float32)

        print(f"✓ Loaded MP3 as numpy array")
        print(f"  Shape: {audio_array.shape}")
        print(f"  Data type: {audio_array.dtype}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(audio_array) / sample_rate:.2f} seconds")
        print(f"  Value range: {np.min(audio_array):.3f} to {np.max(audio_array):.3f}")

        return audio_array, sample_rate

    except ImportError:
        raise ImportError("Install librosa: pip install librosa")


##########################################################################################################################

# import matplotlib.pyplot as plt

# numpy_data = mp3_to_numpy_librosa(r"sample_ukg.mp3")
# arr = analyze_frequencies_sliding_window(numpy_data[0], numpy_data[1], n_bands=8, window_size=1024, hop_size=512)
#
# print(arr[0][0])

# y = np.random.rand(22050)  # 1 second of samples
# time = np.arange(len(y)) / numpy_data[1]  # seconds
# plt.figure(figsize=(15, 3))
# plt.plot(arr[0])
# plt.title("1D Array Visualization")
#
# plt.plot(time, y)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.savefig("plot.png")   # saves to current directory
# print("Saved to plot.png")


bl_info = {
    "name": "Frequency Mapper",
    "author": "Jonah Shin",
    "description": "Map audio frequency to object keyframes",
    "blender": (4, 4, 0),
    "version": (1, 1, 1),
    "location": "View3D > Frequency Mapper",
    "warning": "",
    "category": "Animation"
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