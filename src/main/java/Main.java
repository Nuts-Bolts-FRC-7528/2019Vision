/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import com.google.gson.*;
import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.vision.VisionThread;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static edu.wpi.first.wpilibj.smartdashboard.SmartDashboard.getEntry;

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
    public static class CargoPipeline implements VisionPipeline {
        //Outputs
        private Mat hsvThresholdOutput = new Mat();
        private Mat cvErodeOutput = new Mat();
        private ArrayList<MatOfPoint> findContoursOutput = new ArrayList<MatOfPoint>();
        private ArrayList<MatOfPoint> filterContoursOutput = new ArrayList<MatOfPoint>();
        private ArrayList<MatOfPoint> convexHullsOutput = new ArrayList<MatOfPoint>();

        static {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        }

        /**
         * This is the primary method that runs the entire pipeline and updates the outputs.
         */
        public void process(Mat source0) {
            // Step HSV_Threshold0:
            Mat hsvThresholdInput = source0;
            double[] hsvThresholdHue = {0.0, 47.512365169462136};
            double[] hsvThresholdSaturation = {146.97851887981145, 255.0};
            double[] hsvThresholdValue = {137.69789253343086, 255.0};
            hsvThreshold(hsvThresholdInput, hsvThresholdHue, hsvThresholdSaturation, hsvThresholdValue, hsvThresholdOutput);

            // Step CV_erode0:
            Mat cvErodeSrc = hsvThresholdOutput;
            Mat cvErodeKernel = new Mat();
            Point cvErodeAnchor = new Point(-1, -1);
            double cvErodeIterations = 1;
            int cvErodeBordertype = Core.BORDER_CONSTANT;
            Scalar cvErodeBordervalue = new Scalar(-1);
            cvErode(cvErodeSrc, cvErodeKernel, cvErodeAnchor, cvErodeIterations, cvErodeBordertype, cvErodeBordervalue, cvErodeOutput);

            // Step Find_Contours0:
            Mat findContoursInput = cvErodeOutput;
            boolean findContoursExternalOnly = false;
            findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);

            // Step Filter_Contours0:
            ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
            double filterContoursMinArea = 60.0;
            double filterContoursMinPerimeter = 0;
            double filterContoursMinWidth = 0;
            double filterContoursMaxWidth = 1000;
            double filterContoursMinHeight = 0;
            double filterContoursMaxHeight = 1000;
            double[] filterContoursSolidity = {0, 100};
            double filterContoursMaxVertices = 1000000;
            double filterContoursMinVertices = 16.0;
            double filterContoursMinRatio = 0;
            double filterContoursMaxRatio = 1000;
            filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

            // Step Convex_Hulls0:
            ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
            convexHulls(convexHullsContours, convexHullsOutput);

        }

        /**
         * This method is a generated getter for the output of a HSV_Threshold.
         * @return Mat output from HSV_Threshold.
         */
        public Mat hsvThresholdOutput() {
            return hsvThresholdOutput;
        }

        /**
         * This method is a generated getter for the output of a CV_erode.
         * @return Mat output from CV_erode.
         */
        public Mat cvErodeOutput() {
            return cvErodeOutput;
        }

        /**
         * This method is a generated getter for the output of a Find_Contours.
         * @return ArrayList<MatOfPoint> output from Find_Contours.
         */
        public ArrayList<MatOfPoint> findContoursOutput() {
            return findContoursOutput;
        }

        /**
         * This method is a generated getter for the output of a Filter_Contours.
         * @return ArrayList<MatOfPoint> output from Filter_Contours.
         */
        public ArrayList<MatOfPoint> filterContoursOutput() {
            return filterContoursOutput;
        }

        /**
         * This method is a generated getter for the output of a Convex_Hulls.
         * @return ArrayList<MatOfPoint> output from Convex_Hulls.
         */
        public ArrayList<MatOfPoint> convexHullsOutput() {
            return convexHullsOutput;
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
         * Sets the values of pixels in a binary image to their distance to the nearest black pixel.
         * @param input The image on which to perform the Distance Transform.
         */
        private void findContours(Mat input, boolean externalOnly,
                                  List<MatOfPoint> contours) {
            Mat hierarchy = new Mat();
            contours.clear();
            int mode;
            if (externalOnly) {
                mode = Imgproc.RETR_EXTERNAL;
            }
            else {
                mode = Imgproc.RETR_LIST;
            }
            int method = Imgproc.CHAIN_APPROX_SIMPLE;
            Imgproc.findContours(input, contours, hierarchy, mode, method);
        }


        /**
         * Filters out contours that do not meet certain criteria.
         * @param inputContours is the input list of contours
         * @param output is the the output list of contours
         * @param minArea is the minimum area of a contour that will be kept
         * @param minPerimeter is the minimum perimeter of a contour that will be kept
         * @param minWidth minimum width of a contour
         * @param maxWidth maximum width
         * @param minHeight minimum height
         * @param maxHeight maximimum height
         * @param minVertexCount minimum vertex Count of the contours
         * @param maxVertexCount maximum vertex Count
         * @param minRatio minimum ratio of width to height
         * @param maxRatio maximum ratio of width to height
         */
        private void filterContours(List<MatOfPoint> inputContours, double minArea,
                                    double minPerimeter, double minWidth, double maxWidth, double minHeight, double
                                            maxHeight, double[] solidity, double maxVertexCount, double minVertexCount, double
                                            minRatio, double maxRatio, List<MatOfPoint> output) {
            final MatOfInt hull = new MatOfInt();
            output.clear();
            //operation
            for (int i = 0; i < inputContours.size(); i++) {
                final MatOfPoint contour = inputContours.get(i);
                final Rect bb = Imgproc.boundingRect(contour);
                if (bb.width < minWidth || bb.width > maxWidth) continue;
                if (bb.height < minHeight || bb.height > maxHeight) continue;
                final double area = Imgproc.contourArea(contour);
                if (area < minArea) continue;
                if (Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true) < minPerimeter) continue;
                Imgproc.convexHull(contour, hull);
                MatOfPoint mopHull = new MatOfPoint();
                mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
                for (int j = 0; j < hull.size().height; j++) {
                    int index = (int)hull.get(j, 0)[0];
                    double[] point = new double[] { contour.get(index, 0)[0], contour.get(index, 0)[1]};
                    mopHull.put(j, 0, point);
                }
                final double solid = 100 * area / Imgproc.contourArea(mopHull);
                if (solid < solidity[0] || solid > solidity[1]) continue;
                if (contour.rows() < minVertexCount || contour.rows() > maxVertexCount)	continue;
                final double ratio = bb.width / (double)bb.height;
                if (ratio < minRatio || ratio > maxRatio) continue;
                output.add(contour);
            }
        }

        /**
         * Compute the convex hulls of contours.
         * @param inputContours The contours on which to perform the operation.
         * @param outputContours The contours where the output will be stored.
         */
        private void convexHulls(List<MatOfPoint> inputContours,
                                 ArrayList<MatOfPoint> outputContours) {
            final MatOfInt hull = new MatOfInt();
            outputContours.clear();
            for (int i = 0; i < inputContours.size(); i++) {
                final MatOfPoint contour = inputContours.get(i);
                final MatOfPoint mopHull = new MatOfPoint();
                Imgproc.convexHull(contour, hull);
                mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
                for (int j = 0; j < hull.size().height; j++) {
                    int index = (int) hull.get(j, 0)[0];
                    double[] point = new double[] {contour.get(index, 0)[0], contour.get(index, 0)[1]};
                    mopHull.put(j, 0, point);
                }
                outputContours.add(mopHull);
            }
        }
    }


    /**
     * HatchPipeline class.
     *
     * <p>An OpenCV pipeline generated by GRIP.
     *
     * @author GRIP
     */
    public static class HatchPipeline implements VisionPipeline {

        //Outputs
        private Mat hsvThresholdOutput = new Mat();
        private ArrayList<MatOfPoint> findContoursOutput = new ArrayList<MatOfPoint>();
        private ArrayList<MatOfPoint> filterContoursOutput = new ArrayList<MatOfPoint>();
        private ArrayList<MatOfPoint> convexHullsOutput = new ArrayList<MatOfPoint>();

        static {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        }

        /**
         * This is the primary method that runs the entire pipeline and updates the outputs.
         */
        public void process(Mat source0) {
            // Step HSV_Threshold0:
            Mat hsvThresholdInput = source0;
            double[] hsvThresholdHue = {25.423728813559322, 180.0};
            double[] hsvThresholdSaturation = {163.27683615819208, 255.0};
            double[] hsvThresholdValue = {77.04776856480918, 255.0};
            hsvThreshold(hsvThresholdInput, hsvThresholdHue, hsvThresholdSaturation, hsvThresholdValue, hsvThresholdOutput);

            // Step Find_Contours0:
            Mat findContoursInput = hsvThresholdOutput;
            boolean findContoursExternalOnly = false;
            findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);

            // Step Filter_Contours0:
            ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
            double filterContoursMinArea = 25.0;
            double filterContoursMinPerimeter = 83.0;
            double filterContoursMinWidth = 12.0;
            double filterContoursMaxWidth = 1000.0;
            double filterContoursMinHeight = 22.0;
            double filterContoursMaxHeight = 1000.0;
            double[] filterContoursSolidity = {0, 100};
            double filterContoursMaxVertices = 1000000.0;
            double filterContoursMinVertices = 0.0;
            double filterContoursMinRatio = 0.0;
            double filterContoursMaxRatio = 1000.0;
            filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

            // Step Convex_Hulls0:
            ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
            convexHulls(convexHullsContours, convexHullsOutput);

        }

        /**
         * This method is a generated getter for the output of a HSV_Threshold.
         * @return Mat output from HSV_Threshold.
         */
        public Mat hsvThresholdOutput() {
            return hsvThresholdOutput;
        }

        /**
         * This method is a generated getter for the output of a Find_Contours.
         * @return ArrayList<MatOfPoint> output from Find_Contours.
         */
        public ArrayList<MatOfPoint> findContoursOutput() {
            return findContoursOutput;
        }

        /**
         * This method is a generated getter for the output of a Filter_Contours.
         * @return ArrayList<MatOfPoint> output from Filter_Contours.
         */
        public ArrayList<MatOfPoint> filterContoursOutput() {
            return filterContoursOutput;
        }

        /**
         * This method is a generated getter for the output of a Convex_Hulls.
         * @return ArrayList<MatOfPoint> output from Convex_Hulls.
         */
        public ArrayList<MatOfPoint> convexHullsOutput() {
            return convexHullsOutput;
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
         * Sets the values of pixels in a binary image to their distance to the nearest black pixel.
         * @param input The image on which to perform the Distance Transform.
         */
        private void findContours(Mat input, boolean externalOnly,
                                  List<MatOfPoint> contours) {
            Mat hierarchy = new Mat();
            contours.clear();
            int mode;
            if (externalOnly) {
                mode = Imgproc.RETR_EXTERNAL;
            }
            else {
                mode = Imgproc.RETR_LIST;
            }
            int method = Imgproc.CHAIN_APPROX_SIMPLE;
            Imgproc.findContours(input, contours, hierarchy, mode, method);
        }


        /**
         * Filters out contours that do not meet certain criteria.
         * @param inputContours is the input list of contours
         * @param output is the the output list of contours
         * @param minArea is the minimum area of a contour that will be kept
         * @param minPerimeter is the minimum perimeter of a contour that will be kept
         * @param minWidth minimum width of a contour
         * @param maxWidth maximum width
         * @param minHeight minimum height
         * @param maxHeight maximimum height
         * @param minVertexCount minimum vertex Count of the contours
         * @param maxVertexCount maximum vertex Count
         * @param minRatio minimum ratio of width to height
         * @param maxRatio maximum ratio of width to height
         */
        private void filterContours(List<MatOfPoint> inputContours, double minArea,
                                    double minPerimeter, double minWidth, double maxWidth, double minHeight, double
                                            maxHeight, double[] solidity, double maxVertexCount, double minVertexCount, double
                                            minRatio, double maxRatio, List<MatOfPoint> output) {
            final MatOfInt hull = new MatOfInt();
            output.clear();
            //operation
            for (int i = 0; i < inputContours.size(); i++) {
                final MatOfPoint contour = inputContours.get(i);
                final Rect bb = Imgproc.boundingRect(contour);
                if (bb.width < minWidth || bb.width > maxWidth) continue;
                if (bb.height < minHeight || bb.height > maxHeight) continue;
                final double area = Imgproc.contourArea(contour);
                if (area < minArea) continue;
                if (Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true) < minPerimeter) continue;
                Imgproc.convexHull(contour, hull);
                MatOfPoint mopHull = new MatOfPoint();
                mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
                for (int j = 0; j < hull.size().height; j++) {
                    int index = (int)hull.get(j, 0)[0];
                    double[] point = new double[] { contour.get(index, 0)[0], contour.get(index, 0)[1]};
                    mopHull.put(j, 0, point);
                }
                final double solid = 100 * area / Imgproc.contourArea(mopHull);
                if (solid < solidity[0] || solid > solidity[1]) continue;
                if (contour.rows() < minVertexCount || contour.rows() > maxVertexCount)	continue;
                final double ratio = bb.width / (double)bb.height;
                if (ratio < minRatio || ratio > maxRatio) continue;
                output.add(contour);
            }
        }

        /**
         * Compute the convex hulls of contours.
         * @param inputContours The contours on which to perform the operation.
         * @param outputContours The contours where the output will be stored.
         */
        private void convexHulls(List<MatOfPoint> inputContours,
                                 ArrayList<MatOfPoint> outputContours) {
            final MatOfInt hull = new MatOfInt();
            outputContours.clear();
            for (int i = 0; i < inputContours.size(); i++) {
                final MatOfPoint contour = inputContours.get(i);
                final MatOfPoint mopHull = new MatOfPoint();
                Imgproc.convexHull(contour, hull);
                mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
                for (int j = 0; j < hull.size().height; j++) {
                    int index = (int) hull.get(j, 0)[0];
                    double[] point = new double[] {contour.get(index, 0)[0], contour.get(index, 0)[1]};
                    mopHull.put(j, 0, point);
                }
                outputContours.add(mopHull);
            }
        }




    }


    /**
     * Finds the minimum and maximum X value of a given ArrayList of MatOfPoints
     * @param contours An ArrayList of a MatOfPoints
     * @return An array, where index 0 is the left (minimum) X, and index 1 is the right (maximum) X value.
     */
    private static int[] findMinAndMaxX(ArrayList<MatOfPoint> contours) {
        int[] minMax = {Integer.MAX_VALUE,Integer.MIN_VALUE}; //Where the first value is the left x and the second is the max X
        for(MatOfPoint contour : contours) {
            for(int i = 0; i < contour.toArray().length; i++) {
                if(contour.toArray()[i].x <minMax[0]) {
                    minMax[0] = (int)contour.toArray()[i].x;
                }
                if(contour.toArray()[i].x > minMax[1]) {
                    minMax[1] = (int)contour.toArray()[i].x;
                }
            }
        }

        return minMax;
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
        NetworkTableEntry isTrackingCargo = SmartDashboard.getEntry("Cargo Tracking:");
        NetworkTableEntry isTrackingHatch = SmartDashboard.getEntry("Hatch Tracking:");
        NetworkTableEntry cargoCenterPix= table.getEntry("cargoCenterPix");
        NetworkTableEntry hatchCenterPix = table.getEntry("hatchCenterPix");
        ntinst.startClientTeam(7528);   
        ntinst.startDSClient();

        // start cameras
        List<VideoSource> cameras = new ArrayList<>();
        for (CameraConfig cameraConfig : cameraConfigs) {
            cameras.add(startCamera(cameraConfig));
        }

        // start image processing on camera 0 if present
        if (cameras.size() >= 1) {
            /*
                        [CARGO]
             */
            VisionThread cargoVisionThread = new VisionThread(cameras.get(0),
                    new CargoPipeline(), pipeline -> {
                ArrayList<MatOfPoint> contours =  pipeline.convexHullsOutput();
                int[] minMax = findMinAndMaxX(contours);

                if(minMax[0] != Integer.MAX_VALUE) { //If the ball IS found
                    isTrackingCargo.setBoolean(true);
                    cargoCenterPix.setDouble((minMax[0] + minMax[1]) / 2.0);

                } else { //If the ball is NOT found
                    isTrackingCargo.setBoolean(false);
                    cargoCenterPix.setDouble(-1);
                }
                System.out.println("Center pixel CARGO: " + ((minMax[0] + minMax[1]) / 2.0));
            });

            /*
                    [HATCH]
             */
            VisionThread hatchVisionThread = new VisionThread(cameras.get(0),
                    new HatchPipeline(), pipeline -> {
                ArrayList<MatOfPoint> contours = pipeline.convexHullsOutput();
                int[] minMax = findMinAndMaxX(contours);

                if(minMax[0] != Integer.MAX_VALUE) {
                    isTrackingHatch.setBoolean(true);
                    hatchCenterPix.setDouble((minMax[0] + minMax[1]) / 2.0);
                } else {
                    isTrackingHatch.setBoolean(false);
                    hatchCenterPix.setDouble(-1);
                }
                System.out.println("Center pixel HATCH: " + ((minMax[0] + minMax[1]) / 2.0));
            });

            cargoVisionThread.start(); //Start cargo thread
            hatchVisionThread.start(); //Start hatch thread
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