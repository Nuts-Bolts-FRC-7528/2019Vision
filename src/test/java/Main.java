/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.vision.VisionThread;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.LineSegmentDetector;

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
   }
 */

public final class Main {
    private static String configFile = "/boot/frc.json";

    @SuppressWarnings("MemberName")
    public static class CameraConfig {
        public String name;
        public String path;
        public JsonObject config;
        public JsonElement streamConfig;
    }

    public static int team;
    public static boolean server;
    public static List<CameraConfig> cameraConfigs = new ArrayList<>();

    private Main() {
    }

    /**
     * Report parse error.
     */
    public static void parseError(String str) {
        System.err.println("config error in '" + configFile + "': " + str);
    }

    /**
     * Read single camera configuration.
     */
    public static boolean readCameraConfig(JsonObject config) {
        CameraConfig cam = new CameraConfig();

        // name
        JsonElement nameElement = config.get("name");
        if (nameElement == null) {
            parseError("could not read camera name");
            return false;
        }
        cam.name = nameElement.getAsString();

        // path
        JsonElement pathElement = config.get("path");
        if (pathElement == null) {
            parseError("camera '" + cam.name + "': could not read path");
            return false;
        }
        cam.path = pathElement.getAsString();

        // stream properties
        cam.streamConfig = config.get("stream");

        cam.config = config;

        cameraConfigs.add(cam);
        return true;
    }

    /**
     * Read configuration file.
     */
    @SuppressWarnings("PMD.CyclomaticComplexity")
    public static boolean readConfig() {
        // parse file
        JsonElement top;
        try {
            top = new JsonParser().parse(Files.newBufferedReader(Paths.get(configFile)));
        } catch (IOException ex) {
            System.err.println("could not open '" + configFile + "': " + ex);
            return false;
        }

        // top level must be an object
        if (!top.isJsonObject()) {
            parseError("must be JSON object");
            return false;
        }
        JsonObject obj = top.getAsJsonObject();

        // team number
        JsonElement teamElement = obj.get("team");
        if (teamElement == null) {
            parseError("could not read team number");
            return false;
        }
        team = teamElement.getAsInt();

        // ntmode (optional)
        if (obj.has("ntmode")) {
            String str = obj.get("ntmode").getAsString();
            if ("client".equalsIgnoreCase(str)) {
                server = false;
            } else if ("server".equalsIgnoreCase(str)) {
                server = true;
            } else {
                parseError("could not understand ntmode value '" + str + "'");
            }
        }

        // cameras
        JsonElement camerasElement = obj.get("cameras");
        if (camerasElement == null) {
            parseError("could not read cameras");
            return false;
        }
        JsonArray cameras = camerasElement.getAsJsonArray();
        for (JsonElement camera : cameras) {
            if (!readCameraConfig(camera.getAsJsonObject())) {
                return false;
            }
        }

        return true;
    }

    /**
     * Start running the camera.
     */
    public static VideoSource startCamera(CameraConfig config) {
        System.out.println("Starting camera '" + config.name + "' on " + config.path);
        CameraServer inst = CameraServer.getInstance();
        UsbCamera camera = new UsbCamera(config.name, config.path);
        MjpegServer server = inst.startAutomaticCapture(camera);

        Gson gson = new GsonBuilder().create();

        camera.setConfigJson(gson.toJson(config.config));
        camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);

        if (config.streamConfig != null) {
            server.setConfigJson(gson.toJson(config.streamConfig));
        }

        return camera;
    }

    /**
     * Example pipeline.
     */
    public static class MyPipeline implements VisionPipeline {
        private Mat hsvThresholdOutput = new Mat();
        private Mat cvBitwiseNotOutput = new Mat();
        private Mat cvErodeOutput = new Mat();
        private Mat maskOutput = new Mat();
        private ArrayList<Line> findLinesOutput = new ArrayList<Line>();

        static {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        }

        /**
         * This is the primary method that runs the entire pipeline and updates the outputs.
         */
        @Override	public void process(Mat source0) {
            // Step HSV_Threshold0:
            Mat hsvThresholdInput = source0;
            double[] hsvThresholdHue = {30.755395683453237, 180.0};
            double[] hsvThresholdSaturation = {0.0, 255.0};
            double[] hsvThresholdValue = {0.0, 255.0};
            hsvThreshold(hsvThresholdInput, hsvThresholdHue, hsvThresholdSaturation, hsvThresholdValue, hsvThresholdOutput);

            // Step CV_bitwise_not0:
            Mat cvBitwiseNotSrc1 = hsvThresholdOutput;
            cvBitwiseNot(cvBitwiseNotSrc1, cvBitwiseNotOutput);

            // Step CV_erode0:
            Mat cvErodeSrc = cvBitwiseNotOutput;
            Mat cvErodeKernel = new Mat();
            Point cvErodeAnchor = new Point(-1, -1);
            double cvErodeIterations = 26.0;
            int cvErodeBordertype = Core.BORDER_CONSTANT;
            Scalar cvErodeBordervalue = new Scalar(-1);
            cvErode(cvErodeSrc, cvErodeKernel, cvErodeAnchor, cvErodeIterations, cvErodeBordertype, cvErodeBordervalue, cvErodeOutput);

            // Step Mask0:
            Mat maskInput = source0;
            Mat maskMask = cvErodeOutput;
            mask(maskInput, maskMask, maskOutput);

            // Step Find_Lines0:
            Mat findLinesInput = maskOutput;
            findLines(findLinesInput, findLinesOutput);

        }

        /**
         * This method is a generated getter for the output of a HSV_Threshold.
         * @return Mat output from HSV_Threshold.
         */
        public Mat hsvThresholdOutput() {
            return hsvThresholdOutput;
        }

        /**
         * This method is a generated getter for the output of a CV_bitwise_not.
         * @return Mat output from CV_bitwise_not.
         */
        public Mat cvBitwiseNotOutput() {
            return cvBitwiseNotOutput;
        }

        /**
         * This method is a generated getter for the output of a CV_erode.
         * @return Mat output from CV_erode.
         */
        public Mat cvErodeOutput() {
            return cvErodeOutput;
        }

        /**
         * This method is a generated getter for the output of a Mask.
         * @return Mat output from Mask.
         */
        public Mat maskOutput() {
            return maskOutput;
        }

        /**
         * This method is a generated getter for the output of a Find_Lines.
         * @return ArrayList<Line> output from Find_Lines.
         */
        public ArrayList<Line> findLinesOutput() {
            return findLinesOutput;
        }


        /**
         * Segment an image based on hue, saturation, and value ranges.
         *
         * @param input The image on which to perform the HSL threshold.
         * @param hue The min and max hue
         * @param sat The min and max saturation
         * @param val The min and max value
         */
        private void hsvThreshold(Mat input, double[] hue, double[] sat, double[] val,
                                  Mat out) {
            Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HSV);
            Core.inRange(out, new Scalar(hue[0], sat[0], val[0]),
                    new Scalar(hue[1], sat[1], val[1]), out);
        }

        /**
         * Computes the per element inverse of an image.
         * @param src the image to invert.
         * @param dst the inversion of the input image.
         */
        private void cvBitwiseNot(Mat src, Mat dst) {
            Core.bitwise_not(src, dst);
        }

        /**
         * Expands area of lower value in an image.
         * @param src the Image to erode.
         * @param kernel the kernel for erosion.
         * @param anchor the center of the kernel.
         * @param iterations the number of times to perform the erosion.
         * @param borderType pixel extrapolation method.
         * @param borderValue value to be used for a constant border.
         * @param dst Output Image.
         */
        private void cvErode(Mat src, Mat kernel, Point anchor, double iterations,
                             int borderType, Scalar borderValue, Mat dst) {
            if (kernel == null) {
                kernel = new Mat();
            }
            if (anchor == null) {
                anchor = new Point(-1,-1);
            }
            if (borderValue == null) {
                borderValue = new Scalar(-1);
            }
            Imgproc.erode(src, dst, kernel, anchor, (int)iterations, borderType, borderValue);
        }

        /**
         * Filter out an area of an image using a binary mask.
         * @param input The image on which the mask filters.
         * @param mask The binary image that is used to filter.
         * @param output The image in which to store the output.
         */
        private void mask(Mat input, Mat mask, Mat output) {
            mask.convertTo(mask, CvType.CV_8UC1);
            Core.bitwise_xor(output, output, output);
            input.copyTo(output, mask);
        }

        public static class Line {
            public final double x1, y1, x2, y2;
            public Line(double x1, double y1, double x2, double y2) {
                this.x1 = x1;
                this.y1 = y1;
                this.x2 = x2;
                this.y2 = y2;
            }
            public double lengthSquared() {
                return Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2);
            }
            public double length() {
                return Math.sqrt(lengthSquared());
            }
            public double angle() {
                return Math.toDegrees(Math.atan2(y2 - y1, x2 - x1));
            }
        }
        /**
         * Finds all line segments in an image.
         * @param input The image on which to perform the find lines.
         * @param lineList The output where the lines are stored.
         */
        private void findLines(Mat input, ArrayList<Line> lineList) {
            final LineSegmentDetector lsd = Imgproc.createLineSegmentDetector();
            final Mat lines = new Mat();
            lineList.clear();
            if (input.channels() == 1) {
                lsd.detect(input, lines);
            } else {
                final Mat tmp = new Mat();
                Imgproc.cvtColor(input, tmp, Imgproc.COLOR_BGR2GRAY);
                lsd.detect(tmp, lines);
            }
            if (!lines.empty()) {
                for (int i = 0; i < lines.rows(); i++) {
                    lineList.add(new Line(lines.get(i, 0)[0], lines.get(i, 0)[1],
                            lines.get(i, 0)[2], lines.get(i, 0)[3]));
                }
            }
        }
    }

    /**
     * Main.
     */
    public static void main(String... args) {
        if (args.length > 0) {
            configFile = args[0];
        }

        // read configuration
        if (!readConfig()) {
            return;
        }

        // start NetworkTables
        NetworkTableInstance ntinst = NetworkTableInstance.getDefault();
        if (server) {
            System.out.println("Setting up NetworkTables server");
            ntinst.startServer();
        } else {
            System.out.println("Setting up NetworkTables client for team " + team);
            ntinst.startClientTeam(team);
        }

        NetworkTable  table = ntinst.getTable("vision");
        NetworkTableEntry centerPix = table.getEntry("centerPix");
        ntinst.startClientTeam(7528);   
        ntinst.startDSClient();

        // start cameras
        List<VideoSource> cameras = new ArrayList<>();
        for (CameraConfig cameraConfig : cameraConfigs) {
            cameras.add(startCamera(cameraConfig));
        }

        // start image processing on camera 0 if present
        if (cameras.size() >= 1) {
            VisionThread visionThread = new VisionThread(cameras.get(0),
                    new MyPipeline(), pipeline -> {
                /*
                * Actual code for doing things goes here
                 */
                double leftX = Integer.MAX_VALUE;
                double rightX = Integer.MIN_VALUE;
                for(int i = 0; i < pipeline.findLinesOutput.size(); i++){
                    double x1 = pipeline.findLinesOutput.get(i).x1;
                    double x2 = pipeline.findLinesOutput.get(i).x2;
                    double x;
                    if(x2 < x1){
                        x = x2;
                    } else {
                        x = x1;
                    }
                    if (x < leftX) {
                        leftX = x;
                    }
                    if(x > rightX) {
                        rightX = x;
                    }

                }
                centerPix.setDouble(((leftX + rightX) / 2)*5);
                System.out.println("Center X: " + (int)((leftX + rightX) / 2)*5);
                System.out.println("Left X: " + (int)leftX*5);
                System.out.println("Right X: " + (int)rightX*5);
            });
      /* something like this for GRIP:
      VisionThread visionThread = new VisionThread(cameras.get(0),
              new GripPipeline(), pipeline -> {
        ...
      });
       */
            visionThread.start();
        }

        // loop forever
        for (;;) {
            try {
                Thread.sleep(10000);
            } catch (InterruptedException ex) {
                return;
            }
        }
    }
}