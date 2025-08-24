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
        object_scales = []
        for i in range(1,number_value+1):

            obj = bpy.data.objects[str(i)]

            # TEMP!!!########################################################################
            # bpy.ops.mesh.primitive_cube_add()
            # cube = bpy.context.active_object
            # cube.name = str(i)
            # cube.location = (i*2, 0.0, 0.0)
            #################################################################################

            objects.append(obj)
            objects[-1].animation_data_clear()
            object_scales.append(obj.scale.copy())

        epsilon = 1e-6  # for log adjustment
        for i in range(arr_len):
            for j, obj in enumerate(objects):
                # Apply log scaling
                amp = arr[0][i][j]
                log_amp = np.log1p(amp + epsilon)  # log(1 + x)
                z_scale = log_amp * 10.0

                # obj.scale.z = z_scale
                x = object_scales[j][0] + z_scale
                y = object_scales[j][1] + z_scale
                z = object_scales[j][2]
                obj.scale = (x, y, z)
                obj.keyframe_insert(data_path="scale", frame=frames[i])

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

    # Calculate number of time frames
    print(f"len(audio_data)={len(audio_data)}, window_size={window_size}, hop_size={hop_size}")
    n_frames = (len(audio_data) - window_size) // hop_size + 1

    # Set up frequency bands
    min_freq = 20
    max_freq = min(sample_rate // 2, 20000)
    log_min = np.log10(min_freq)
    log_max = np.log10(max_freq)
    band_edges = np.logspace(log_min, log_max, n_bands + 1)

    print(f"n_frames={n_frames}")

    # Initialize output arrays
    time_frequency_data = np.zeros((n_frames, n_bands))
    time_points = np.zeros(n_frames)
    band_ranges = []

    # Create frequency array for window_size
    frequencies = np.fft.fftfreq(window_size, 1 / sample_rate)[:window_size // 2]

    # Calculate band ranges
    for i in range(n_bands):
        band_ranges.append((band_edges[i], band_edges[i + 1]))

    # Process each time frame
    for frame_idx in range(n_frames):
        start_idx = frame_idx * hop_size
        end_idx = start_idx + window_size

        # Extract window
        window = audio_data[start_idx:end_idx]

        # Apply window function (Hanning window)
        windowed = window * np.hanning(window_size)

        # FFT
        fft_data = np.fft.fft(windowed)
        fft_magnitude = np.abs(fft_data[:window_size // 2])

        # Calculate time point
        time_points[frame_idx] = start_idx / sample_rate

        # Extract frequency bands
        for band_idx in range(n_bands):
            freq_low = band_edges[band_idx]
            freq_high = band_edges[band_idx + 1]

            # Find frequencies in this band
            mask = (frequencies >= freq_low) & (frequencies < freq_high)

            if np.any(mask):
                band_energy = np.mean(fft_magnitude[mask])
            else:
                band_energy = 0

            time_frequency_data[frame_idx, band_idx] = band_energy

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