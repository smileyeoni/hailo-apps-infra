import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstPbutils', '1.0')
from gi.repository import Gst, GLib, GstPbutils
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_default_parser,
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_helper_pipelines import(
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    FILE_OUTPUT_PIPELINE,
)
from hailo_apps_infra.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback
)

#-----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class

class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        Gst.init(None)
        if parser == None:
            parser = get_default_parser()
        # Call the parent class constructor
        super().__init__(parser, user_data)

        # Read Video's resolution using Discoverer
        input_path = self.options_menu.input
        if input_path and os.path.isfile(input_path):
            # Create Discoverer (Timeout 5 seconds)
            discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)
            # Filename to URI
            uri = Gst.filename_to_uri(os.path.abspath(input_path))
            info = discoverer.discover_uri(uri)
            # Reading Video stream infomation and peak resolution
            for s in info.get_video_streams():
                if isinstance(s, GstPbutils.DiscovererVideoInfo):
                    struct = s.get_caps().get_structure(0)
                    w = struct.get_int("width")[1]
                    h = struct.get_int("height")[1]
                    self.video_width = w
                    self.video_height = h
                    break
        else:
            # fallback: default values
            self.video_width = 1280
            self.video_height = 720

        #self.batch_size = 2
        self.batch_size = 1
        print(f"Using video resolution: {self.video_width}×{self.video_height}")

        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        #self.video_width = 1280
        #self.video_height = 720


        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch



        # Set the HEF file path based on the architecture
        if self.options_menu.hef_path:
            self.hef_path = self.options_menu.hef_path
        elif self.arch == "hailo8":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8m_pose.hef')
        else:  # hailo8l
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_pose_h8l.hef')

        self.app_callback = app_callback

        # Set the post-processing shared object file
        self.post_process_so = os.path.join(self.current_path, '../resources/libyolov8pose_postprocess.so')
        self.post_process_function = "filter_letterbox"


        # Set the process title
        #setproctitle.setproctitle("Hailo Pose Estimation App")
        setproctitle.setproctitle("BrightMinds Pose Estimation Test")

        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source, video_width=self.video_width, video_height=self.video_height)
        infer_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size
        )
        infer_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(infer_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=0)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        pipeline_string = (
            f'{source_pipeline} !'
            f'{infer_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{user_callback_pipeline} ! '
        )

        # Changed to save file
        if self.file_output:
            output_pipeline = FILE_OUTPUT_PIPELINE("results.mp4")
            pipeline_string += (
                f'{output_pipeline}'
            )
        else:
            display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
            pipeline_string += (
                f'{display_pipeline}'
            )

        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app = GStreamerPoseEstimationApp(dummy_callback, user_data)
    app.run()
