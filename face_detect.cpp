/*
 * MIT Licence.
 * Written by Milind Deore <tomdeore@gmail.com>
 *
 * Mediapipe tensroflow lite face detector model inference using c++ in
 * standlone setup.
 *
 */

#include <cstdio>
#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define LGT_FACE_DETECTION_MODEL_SIZE            (128)
#define LGT_IMAGE_NORM_STD                       (127.5)
#define LGT_IMAGE_NORM_MEAN                      (127.5)

/* Debugging: Enable = 1, disable = 0 */
#define LGT_DEBUG                                (0)

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/*
 * Display any vector content on to the console. 
 */ 
template<typename T>
void lgt_print_vec(std::string vname, std::vector<T> const &a) 
{
    std::cout << vname;
    for(auto i=0; i < a.size(); i++)
    {
        std::cout << a.at(i) << " ";
    }
    std::cout << std::endl;
}

/*
 * Sort vector based on the given indices.
 */ 
std::vector<float> 
lgt_sort_vec(std::vector<float> const &a, std::vector<int> indices)
{
    std::vector<float> retVec;

    for (auto i : indices)
    {
        retVec.push_back(a.at(i));
    }
    return retVec;
}

/*
 * Vector Sclicer.
 * example: 
 * v = [0, 1, 2, 3, 4, 5]
 * slice(v, 0, 3) --> 0, 1, 2.
 * slice(v, 2, 5) --> 2, 3, 4.
 * slice(v, 0, v.size()) --> 0, 1, 2, 3, 4, 5.
 * slice(v, 0, v.size()-1) --> 0, 1, 2, 3, 4.
 */
template<typename T>
std::vector<T> lgt_slice(std::vector<T> const &v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n;

    std::vector<T> vec(first, last);
    return vec;
}

/*
 * Create vector of max values.
 */ 
std::vector<float> 
lgt_vec_maximum(std::vector<float> const &a, std::vector<int> indices)
{
    float compare = 0.0;
    std::vector<float> retVec;

    if (!indices.empty())
    {
        compare = a.at(indices.back());
        std::vector<int> indicesSlice = lgt_slice(indices, 0, indices.size()-1);
        retVec = lgt_sort_vec(a, indicesSlice);
    }
    else
    {
        retVec = a;
    }

    for(int i = 0; i < retVec.size(); i++)
    {
        retVec.at(i) = retVec.at(i) > compare ? retVec.at(i) : compare;
    }
    return retVec;
}


/*
 * Create vector of min values, comparing with max values.
 */ 
std::vector<float> 
lgt_vec_minimum(std::vector<float> const &a, std::vector<int> indices)
{
    float compare = 0.0;
    std::vector<float> retVec;

    if (!indices.empty())
    {
        compare = a.at(indices.back());
        std::vector<int> indicesSlice = lgt_slice(indices, 0, indices.size()-1);
        retVec = lgt_sort_vec(a, indicesSlice);
    }
    else
    {
        retVec = a;
    }

    for(int i = 0; i < retVec.size(); i++)
    {
        retVec.at(i) = retVec.at(i) < compare ? retVec.at(i) : compare;
    }
    return retVec;
}


/*
 * Intersection Over Union (IoU).
 */ 
std::vector<float> 
lgt_iou(std::vector<float> const &a, std::vector<float> const &inter,
        std::vector<int> indices)
{
    float largest = a.at(indices.back());
    indices.pop_back();
    std::vector<float> retVec = lgt_sort_vec(a, indices);
    std::vector<float> totalArea(retVec.size());

    std::transform(retVec.begin(), retVec.end(), totalArea.begin(), bind2nd(std::plus<float>(), largest));
    std::vector<float> iou(inter.size());
    std::transform(totalArea.begin(), totalArea.end(), inter.begin(), iou.begin(), std::minus<float>());
    std::transform(inter.begin(), inter.end(), iou.begin(), iou.begin(), std::divides<float>());

    return iou;
}

/*
 * IOU Argsort logic.
 */ 
std::vector<int>
lgt_iou_argsort(std::vector<float> const &a, float threshold)
{
    std::vector<int> vArg;
    uint32_t index = 0;

    for (auto i : a)
    {
        if (a.at(i) <= threshold)
        {
            vArg.push_back(index);
        }
        index++;
    }

    return vArg;
}


/*
 * Generic Argsort
 * Sort based on order (largest or smallest) and return the indexes.
 */ 
template <typename Iter, typename Compare>
std::vector<int> argsort(Iter begin, Iter end, Compare comp)
{
	// Begin Iterator, End Iterator, Comp
	std::vector<std::pair<int, Iter> > pairList; // Pair Vector
	std::vector<int> ret;                        // Will hold the indices

	int i = 0;
	for (auto it = begin; it < end; it++)
	{
		std::pair<int, Iter> pair(i, it); // 0: Element1, 1:Element2...
		pairList.push_back(pair);         // Add to list
		i++;
	}
	// Stable sort the pair vector
	std::stable_sort(pairList.begin(), pairList.end(),
		[comp](std::pair<int, Iter> prev, std::pair<int, Iter> next) -> bool
	{return comp(*prev.second, *next.second); }  // This is the important part explained below
	);

	for (auto i : pairList)
		ret.push_back(i.first);

	return ret;
}


class SsdAnchorsCalculatorOptions
{
    public:

        // Size of input images.
        uint16_t input_size_width;
        uint16_t input_size_height;

        // Min and max scales for generating anchor boxes on feature maps.
        float min_scale;
        float max_scale;

        // The offset for the center of anchors. The value is in the scale of stride.
        // E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
        float anchor_offset_x;
        float anchor_offset_y;

        // List of different aspect ratio to generate anchors.
        float aspect_ratios[1] = {1.0}; 

        // An additional anchor is added with this aspect ratio and a scale
        // interpolated between the scale for a layer and the scale for the next layer
        // (1.0 for the last layer). This anchor is not included if this value is 0.
        float interpolated_scale_aspect_ratio;

        // A boolean to indicate whether the fixed 3 boxes per location is used in the lowest layer.
        bool reduce_boxes_in_lowest_layer = false; 

        // Whether use fixed width and height (e.g. both 1.0f) for each anchor.
        // This option can be used when the predicted anchor width and height are in  pixels.
        bool fixed_anchor_size = false;

        // Sizes of output feature maps to create anchors. Either feature_map size or
        // stride should be provided.
        uint32_t feature_map_width[0];
        uint32_t feature_map_height[0];
        uint32_t feature_map_width_size = sizeof(feature_map_width);
        uint32_t feature_map_height_size = sizeof(feature_map_height);

        // Strides of each output feature maps.
        uint8_t strides[4] = {8, 16, 16, 16};
        uint8_t strides_size;

        // Number of output feature maps to generate the anchors on.
        uint8_t num_layers;

        // Sizeof aspect ratio to generate anchors.
        uint8_t aspect_ratios_size;

        std::string to_str()
        {
            std::string retstr;

            retstr += "input_size_width: " + std::to_string(this->input_size_width) + "\n";
            retstr += "input_size_height: " + std::to_string(this->input_size_height) + "\n";
            retstr += "min_scale: " + std::to_string(this->min_scale) + "\n";
            retstr += "max_scale: " + std::to_string(this->max_scale) + "\n";
            retstr += "anchor_offset_x: " + std::to_string(this->anchor_offset_x) + "\n";
            retstr += "anchor_offset_y: " + std::to_string(this->anchor_offset_y) + "\n";
            retstr += "num_layers: " + std::to_string(this->num_layers) + "\n";
            retstr += "feature_map_width: [" + std::to_string(this->feature_map_width[0]) + "]\n";
            retstr += "feature_map_height: [" + std::to_string(this->feature_map_height[0]) + "]\n";
            retstr += "strides: [" + std::to_string(this->strides[0]) + " " + 
                                     std::to_string(this->strides[1]) + " " +
                                     std::to_string(this->strides[2]) + " " +
                                     std::to_string(this->strides[3]) + " " + "]\n";
            retstr += "aspect_ratios: " + std::to_string(this->aspect_ratios[0]) + "\n";
            retstr += "reduce_boxes_in_lowest_layer: " + std::to_string(this->reduce_boxes_in_lowest_layer) + "\n";
            retstr += "interpolated_scale_aspect_ratio: " + std::to_string(this->interpolated_scale_aspect_ratio) + "\n";
            retstr += "fixed_anchor_size: " + std::to_string(this->fixed_anchor_size) + "\n";
            return retstr;
        }

        SsdAnchorsCalculatorOptions(uint16_t input_size_width, uint16_t input_size_height, float min_scale, float max_scale,
                                    float anchor_offset_x, float anchor_offset_y, float interpolated_scale_aspect_ratio, 
                                    bool reduce_boxes_in_lowest_layer, bool fixed_anchor_size, uint8_t num_layers)
        {
            this->input_size_width = input_size_width;
            this->input_size_height = input_size_height;
            this->min_scale = min_scale;
            this->max_scale = max_scale;
            this->anchor_offset_x = anchor_offset_x ? anchor_offset_x : 0.5;
            this->anchor_offset_y = anchor_offset_y ? anchor_offset_y : 0.5;
            this->aspect_ratios_size = sizeof(this->aspect_ratios) / sizeof(this->aspect_ratios[0]);
            this->interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio ? interpolated_scale_aspect_ratio : 1.0;
            this->reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer ? reduce_boxes_in_lowest_layer : false;
            this->fixed_anchor_size = fixed_anchor_size ? fixed_anchor_size : false;
            this->strides_size = sizeof(this->strides);
            this->num_layers = num_layers;
        }

        ~SsdAnchorsCalculatorOptions() {};
};


class Anchor
{
    public:
        float x_center;
        float y_center;
        float h;
        float w;

        Anchor()
        {
            this->x_center = 0;
            this->y_center = 0;
            this->h = 0;
            this->w = 0;
        }

        std::string to_str()
        {
            std::string retstr;

            retstr += "x_center : " + std::to_string(this->x_center) + "\n";
            retstr += "y_center : " + std::to_string(this->y_center) + "\n";
            retstr += "h : " + std::to_string(this->h) + "\n";
            retstr += "w : " + std::to_string(this->w) + "\n";
            return retstr;
        }

        ~Anchor() {};
};

class Detection
{
    public:
        float score;
        float class_id;
        float xmin;
        float ymin;
        float width;
        float height;

        Detection()
        {
            this->score = 0.0;
            this->class_id = 0.0;
            this->xmin = 0.0;
            this->ymin = 0.0;
            this->width = 0.0;
            this->height = 0.0;
        }
        
        Detection(float score, float class_id, float xmin, float ymin, float width, float height)
        {
            this->score = score;
            this->class_id = class_id;
            this->xmin = xmin;
            this->ymin = ymin;
            this->width = width;
            this->height = height;
        }

        std::string to_str()
        {
            std::string retstr;

            retstr += "score : " + std::to_string(this->score) + "\n";
            retstr += "class_id : " + std::to_string(this->class_id) + "\n";
            retstr += "xmin : " + std::to_string(this->xmin) + "\n";
            retstr += "ymin : " + std::to_string(this->ymin) + "\n";
            retstr += "width : " + std::to_string(this->width) + "\n";
            retstr += "height : " + std::to_string(this->height) + "\n";
            return retstr;
        }
        
        ~Detection() {};
};

class TfLiteTensorsToDetectionsCalculatorOptions
{
    public:

        uint32_t num_classes;
        uint32_t num_boxes;
        uint32_t num_coords;
        uint32_t keypoint_coord_offset;
        uint32_t num_keypoints;
        uint32_t num_values_per_keypoint;
        uint32_t box_coord_offset;
        float x_scale;
        float y_scale;
        float w_scale;
        float h_scale;
        float score_clipping_thresh;
        float min_score_thresh;
        bool apply_exponential_on_box_size;
        bool reverse_output_order;
        bool sigmoid_score;
        bool flip_vertically;

        TfLiteTensorsToDetectionsCalculatorOptions(uint32_t num_classes, uint32_t num_boxes, uint32_t num_coords, uint32_t keypoint_coord_offset, float score_clipping_thresh, float min_score_thresh, uint32_t num_keypoints, uint32_t num_values_per_keypoint, uint32_t box_coord_offset, float x_scale, float y_scale, float w_scale, float h_scale, bool apply_exponential_on_box_size, bool reverse_output_order, bool sigmoid_score, bool flip_vertically)
        {
            this->num_classes = num_classes;
            this->num_boxes = num_boxes;
            this->num_coords = num_coords;
            this->keypoint_coord_offset = keypoint_coord_offset;
            this->num_keypoints = num_keypoints ? num_keypoints : 0;
            this->num_values_per_keypoint = num_values_per_keypoint ? num_values_per_keypoint : 2;
            this->box_coord_offset = box_coord_offset ? box_coord_offset : 0;
            this->x_scale = x_scale ? x_scale : 0.0;
            this->y_scale = y_scale ? y_scale : 0.0;
            this->w_scale = w_scale ? w_scale : 0.0;
            this->h_scale = h_scale ? h_scale : 0.0;
            this->score_clipping_thresh = score_clipping_thresh;
            this->min_score_thresh = min_score_thresh;
            this->apply_exponential_on_box_size = apply_exponential_on_box_size ? apply_exponential_on_box_size : false;
            this->reverse_output_order = reverse_output_order ? reverse_output_order : false;
            this->sigmoid_score = sigmoid_score ? sigmoid_score : false;
            this->flip_vertically = flip_vertically ? flip_vertically : false;
        }

        std::string to_str()
        {
            std::string retstr;

            retstr += "num_classes : " + std::to_string(this->num_classes) + "\n";
            retstr += "num_boxes : " + std::to_string(this->num_boxes) + "\n";
            retstr += "num_coords : " + std::to_string(this->num_coords) + "\n";
            retstr += "keypoint_coord_offset : " + std::to_string(this->keypoint_coord_offset) + "\n";
            retstr += "num_keypoints : " + std::to_string(this->num_keypoints) + "\n";
            retstr += "num_values_per_keypoint : " + std::to_string(this->num_values_per_keypoint) + "\n";
            retstr += "box_coord_offset : " + std::to_string(this->box_coord_offset) + "\n";
            retstr += "x_scale : " + std::to_string(this->x_scale) + "\n";
            retstr += "y_scale : " + std::to_string(this->y_scale) + "\n";
            retstr += "w_scale : " + std::to_string(this->w_scale) + "\n";
            retstr += "h_scale : " + std::to_string(this->h_scale) + "\n";
            retstr += "score_clipping_thresh : " + std::to_string(this->score_clipping_thresh) + "\n";
            retstr += "min_score_thresh : " + std::to_string(this->min_score_thresh) + "\n";
            retstr += "apply_exponential_on_box_size : " + std::to_string(this->apply_exponential_on_box_size) + "\n";
            retstr += "reverse_output_order : " + std::to_string(this->reverse_output_order) + "\n";
            retstr += "sigmoid_score : " + std::to_string(this->sigmoid_score) + "\n";
            retstr += "flip_vertically : " + std::to_string(this->flip_vertically) + "\n";
            return retstr;
        }

        ~TfLiteTensorsToDetectionsCalculatorOptions() {};
};


std::vector<float> 
lgt_decode_box(std::vector<float> raw_boxes, std::vector<Anchor> anchors,
               TfLiteTensorsToDetectionsCalculatorOptions options,
               uint32_t idx)
{
    std::vector<float> box_data(options.num_coords, 0.0);
        
    uint32_t box_offset = idx * options.num_coords + options.box_coord_offset;

    float y_center = raw_boxes[box_offset];
    float x_center = raw_boxes[box_offset + 1];
    float h = raw_boxes[box_offset + 2];
    float w = raw_boxes[box_offset + 3];

    if (options.reverse_output_order)
    {
        x_center = raw_boxes[box_offset];
        y_center = raw_boxes[box_offset + 1];
        w = raw_boxes[box_offset + 2];
        h = raw_boxes[box_offset + 3];
    }

    x_center = x_center / options.x_scale * anchors[idx].w + anchors[idx].x_center;
    y_center = y_center / options.y_scale * anchors[idx].h + anchors[idx].y_center;

    if (options.apply_exponential_on_box_size)
    {
        h = exp(h / options.h_scale) * anchors[idx].h;
        w = exp(w / options.w_scale) * anchors[idx].w;
    }
    else
    {
        h = h / options.h_scale * anchors[idx].h;
        w = w / options.w_scale * anchors[idx].w;
    }

    float ymin = y_center - h / 2.0;
    float xmin = x_center - w / 2.0;
    float ymax = y_center + h / 2.0;
    float xmax = x_center + w / 2.0;

    box_data[0] = ymin;
    box_data[1] = xmin;
    box_data[2] = ymax;
    box_data[3] = xmax;

    if (options.num_keypoints)
    {
        for (uint32_t k = 0; k < options.num_keypoints; k++)
        {
            uint32_t offset = idx * options.num_coords + options.keypoint_coord_offset + k * options.num_values_per_keypoint;

            float keypoint_y = raw_boxes[offset];
            float keypoint_x = raw_boxes[offset + 1];
            if (options.reverse_output_order)
            {
                keypoint_x = raw_boxes[offset];
                keypoint_y = raw_boxes[offset + 1];
            }

            box_data[4 + k * options.num_values_per_keypoint] = keypoint_x / options.x_scale * anchors[idx].w + anchors[idx].x_center;
            box_data[4 + k * options.num_values_per_keypoint  + 1] = keypoint_y / options.y_scale * anchors[idx].h + anchors[idx].y_center;
        }
    }

    return box_data;
}


Detection 
lgt_convert_to_detection(float box_ymin, float box_xmin, float box_ymax, float box_xmax, 
                         float score, float class_id, bool flip_vertically) 
{
    Detection detection = Detection(score, class_id, box_xmin, (flip_vertically ? 1.0 - box_ymax : box_ymin), (box_xmax - box_xmin), (box_ymax - box_ymin));

    return detection;
}


std::vector<Detection> 
lgt_convert_to_detections(std::vector<float> raw_boxes, std::vector<Anchor> anchors_, 
                          std::vector<float> detection_scores, std::vector<float> detection_classes, 
                          TfLiteTensorsToDetectionsCalculatorOptions options)
{
    std::vector<Detection> output_detections;

    for (int i = 0; i < options.num_boxes; i++)
    {
        if (detection_scores[i] < options.min_score_thresh)
        {
            continue;
        }
        uint32_t box_offset = 0;
        std::vector<float> box_data = lgt_decode_box(raw_boxes, anchors_, options, i);
        Detection detection = lgt_convert_to_detection(
                box_data[box_offset + 0], box_data[box_offset + 1],
                box_data[box_offset + 2], box_data[box_offset + 3],
                detection_scores[i], detection_classes[i], options.flip_vertically);
        
        output_detections.push_back(detection);
    }
    return output_detections;
}


/*
 * Postprocessing on CPU for model without postprocessing op. E.g. output
 * raw score tensor and box tensor. Anchor decoding will be handled below.
 */ 
std::vector<Detection> 
lgt_process_cpu(std::vector<float> raw_boxes, std::vector<float> raw_scores,
                std::vector<Anchor> anchors, 
                TfLiteTensorsToDetectionsCalculatorOptions options)
{
    std::vector<float> detection_scores(options.num_boxes, 0.0);
    std::vector<float> detection_classes(options.num_boxes, 0.0);
    std::vector<Detection> output_detections;

    // Filter classes by scores.
    for (int i = 0; i < options.num_boxes; i++)
    {
        int class_id = -1;
        float max_score = std::numeric_limits<float>::min();

        // Find the top score for box i.
        for (int score_idx = 0; score_idx < options.num_classes; score_idx++)
        {
            float score = raw_scores[i * options.num_classes + score_idx];
            if (options.sigmoid_score)
            {
                if (options.score_clipping_thresh > 0)
                {
                    score = score < -options.score_clipping_thresh ? -options.score_clipping_thresh : score;
                    score = score > options.score_clipping_thresh ? options.score_clipping_thresh : score;
                }
                score = 1.0 / (1.0 + exp(-score));
            }
            if (max_score < score)
            {
                max_score = score;
                class_id = score_idx;
            }
        }
        detection_scores[i] = max_score;
        detection_classes[i] = class_id;
    }

    //cout << "--------------------------------" << endl;
    //cout << "boxes: " << endl;
    //cout << "(" << raw_boxes.size() << ",)" <<endl;
    //lgt_print_vec("", raw_boxes);
    //cout << "--------------------------------" << endl;
    //cout << "detection_scores: " << endl;
    //cout << "(" << detection_scores.size() << ",)" <<endl;
    //lgt_print_vec("", detection_scores);
    //cout << "--------------------------------" << endl;
    //cout << "detection_classes: " << endl;
    //cout << "(" << detection_classes.size() << ",)" <<endl;
    //lgt_print_vec("", detection_classes);

    output_detections = lgt_convert_to_detections(raw_boxes, anchors, detection_scores, 
                                                  detection_classes, options);
    return output_detections;
}


std::vector<Detection> 
lgt_orig_nms(std::vector<Detection> detections, float threshold)
{
    if (detections.size() <= 0)
    {
        return std::vector<Detection>();
    }

    std::vector<float> x1;
    std::vector<float> x2;
    std::vector<float> y1;
    std::vector<float> y2;
    std::vector<float> s;

    for (std::vector<Detection>::iterator ptr = detections.begin(); ptr < detections.end(); ptr++)
    {
        x1.push_back(ptr->xmin);
        x2.push_back(ptr->xmin + ptr->width);
        y1.push_back(ptr->ymin);
        y2.push_back(ptr->ymin + ptr->height);
        s.push_back(ptr->score);
    }
    
    std::vector<float> X(x1.size());
    std::vector<float> Y(y1.size());
    std::vector<float> area(x1.size());

    std::transform(x2.begin(), x2.end(), x1.begin(), X.begin(), std::minus<float>());
    std::transform(X.begin(), X.end(), X.begin(), bind2nd(std::plus<float>(), 1));

    std::transform(y2.begin(), y2.end(), y1.begin(), Y.begin(), std::minus<float>());
    std::transform(Y.begin(), Y.end(), Y.begin(), bind2nd(std::plus<float>(), 1));

    std::transform(X.begin(), X.end(), Y.begin(), area.begin(), std::multiplies<float>());
     
    std::vector<int> indices = argsort(s.begin(), s.end(), std::less<float>());

    std::vector<float> pick;
    while (indices.size() > 0)
    {
        std::vector<float> xx1 = lgt_vec_maximum(x1, indices);   
        std::vector<float> yy1 = lgt_vec_maximum(y1, indices);   
        std::vector<float> xx2 = lgt_vec_minimum(x2, indices);   
        std::vector<float> yy2 = lgt_vec_minimum(y2, indices);   

        std::vector<float> XX(xx1.size());
        std::vector<float> YY(yy1.size());

        std::transform(xx2.begin(), xx2.end(), xx1.begin(), XX.begin(), std::minus<float>());
        std::transform(XX.begin(), XX.end(), XX.begin(), bind2nd(std::plus<float>(), 1.0));

        std::transform(yy2.begin(), yy2.end(), yy1.begin(), YY.begin(), std::minus<float>());
        std::transform(YY.begin(), YY.end(), YY.begin(), bind2nd(std::plus<float>(), 1.0));

        std::vector<float> w = lgt_vec_maximum(XX, std::vector<int>());
        std::vector<float> h = lgt_vec_maximum(YY, std::vector<int>());

        std::vector<float> inter(w.size());
        std::transform(w.begin(), w.end(), h.begin(), inter.begin(), std::multiplies<float>());

        std::vector<float> iou = lgt_iou(area, inter, indices);
        pick.push_back(indices.back());

        indices = lgt_iou_argsort(iou, threshold);
    }

    std::vector<Detection> retDetections;

    for (int i = 0; i < pick.size(); i++)
    {
        retDetections.push_back(detections.at(pick[i]));
    }
    return retDetections;

}

#define NEG(v)          (-(v < 0))

std::vector<Anchor> 
lgt_gen_anchors(SsdAnchorsCalculatorOptions options)
{
    std::vector<Anchor> anchors;

    // Verify the options.
    if (options.strides_size != options.num_layers)
    {
        std::cout << "strides_size and num_layers must be equal." <<std::endl;
        return anchors;
    }

    uint32_t layer_id = 0;
    while (layer_id < options.strides_size)
    {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;

        // For same strides, we merge the anchors in the same order.
        uint32_t last_same_stride_layer = layer_id;
        while (last_same_stride_layer < options.strides_size && options.strides[last_same_stride_layer] == options.strides[layer_id])
        {
            float scale = options.min_scale + (options.max_scale - options.min_scale) * 1.0 * last_same_stride_layer / (options.strides_size - 1.0);
            if (last_same_stride_layer == 0 && options.reduce_boxes_in_lowest_layer)
            {
                // For first layer, it can be specified to use predefined anchors.
                aspect_ratios.push_back(1.0);
                aspect_ratios.push_back(2.0);
                aspect_ratios.push_back(0.5);
                scales.push_back(0.1);
                scales.push_back(scale);
                scales.push_back(scale);
            }
            else
            {
                for (size_t aspect_ratio_id = 0; aspect_ratio_id < options.aspect_ratios_size; aspect_ratio_id++)
                {
                    aspect_ratios.push_back(options.aspect_ratios[aspect_ratio_id]);
                    scales.push_back(scale);
                }

                if (options.interpolated_scale_aspect_ratio > 0.0)
                {
                    float scale_next = (last_same_stride_layer == options.strides_size - 1) ? 1.0 : (options.min_scale + (options.max_scale - options.min_scale) * 1.0 * (last_same_stride_layer + 1) / (options.strides_size - 1.0));
                    scales.push_back(sqrt(scale * scale_next));
                    aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer += 1;
        }
        
        for (size_t i = 0; i < aspect_ratios.size(); i++)
        {
            float ratio_sqrts = sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        uint32_t feature_map_height = 0;
        uint32_t feature_map_width = 0;
        if (options.feature_map_height_size > 0)
        {
            feature_map_height = options.feature_map_height[layer_id];
            feature_map_width = options.feature_map_width[layer_id];
        }
        else
        {
            uint32_t stride = options.strides[layer_id];
            feature_map_height = ceil(1.0 * options.input_size_height / stride);
            feature_map_width = ceil(1.0 * options.input_size_width / stride);
        }

        for (size_t y = 0; y < feature_map_height; y++)
        {
            for (size_t x = 0; x < feature_map_width; x++)
            {
                for (uint32_t anchor_id = 0; anchor_id < anchor_height.size(); anchor_id++)
                {
                    float x_center = (x + options.anchor_offset_x) * 1.0 / feature_map_width;
                    float y_center = (y + options.anchor_offset_y) * 1.0 / feature_map_height;
                    float w = 0;
                    float h = 0;
                    if (options.fixed_anchor_size)
                    {
                        w = 1.0;
                        h = 1.0;
                    }
                    else
                    {
                        w = anchor_width[anchor_id];
                        h = anchor_height[anchor_id];
                    }
                    Anchor new_anchor;
                    new_anchor.x_center = x_center;
                    new_anchor.y_center = y_center;
                    new_anchor.h = h;
                    new_anchor.w = w;
                    anchors.push_back(new_anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    return anchors;
}


int main(int argc, char* argv[]) {

    const char* filename = "./models/face_detection_front.tflite";

    SsdAnchorsCalculatorOptions ssd_anchors_calculator_options(128, 128, 0.1484375, 0.75, 0.5, 0.5, 1.0, false, true, 4);

#if LGT_DEBUG
    cout << "------------------------------------------------" << endl;
    cout << "SsdAnchorsCalculatorOptions: " << endl;
    cout << ssd_anchors_calculator_options.to_str() << endl;
#endif

    std::vector<Anchor> anchors = lgt_gen_anchors(ssd_anchors_calculator_options);

#if LGT_DEBUG
    cout << "------------------------------------------------" << endl;
    cout << "Anchors: " << endl;
    cout << "number: " << anchors.size() << endl;

    for (int i = 0; i < anchors.size(); i++)
    {
        cout << "Anchor " << i << endl;
        cout << anchors[i].to_str() << endl;
    }
#endif

    TfLiteTensorsToDetectionsCalculatorOptions options(1, 896, 16, 4, 100.0, 0.75, 6, 2, 0, 128.0, 128.0, 128.0, 128.0, false, true, true, false);

#if LGT_DEBUG
    cout << "------------------------------------------------" << endl;
    cout << "TfLiteTensorsToDetectionsCalculatorOptions: " << endl;
    cout << options.to_str() << endl;
#endif

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
       tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");
    //tflite::PrintInterpreterState(interpreter.get());

    // Fill input buffers
    int input = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input);
    float *dst = input_tensor->data.f;

    // Read output buffers
    // 1. Regressor
    int reg_output = interpreter->outputs()[0];
    TfLiteTensor* reg_output_tensor = interpreter->tensor(reg_output);
    TfLiteIntArray* reg_output_dims = interpreter->tensor(reg_output)->dims;
    int reg_rows = reg_output_dims->data[reg_output_dims->size - 2];
    int reg_cols = reg_output_dims->data[reg_output_dims->size - 1];
    float *regressors = reg_output_tensor->data.f;
    std::vector<float> regressorVec(reg_rows * reg_cols);

    // 2. Classificator
    int cls_output = interpreter->outputs()[1];
    TfLiteTensor* cls_output_tensor = interpreter->tensor(cls_output);
    TfLiteIntArray* cls_output_dims = interpreter->tensor(cls_output)->dims;
    int cls_rows = cls_output_dims->data[cls_output_dims->size - 2];
    int cls_cols = cls_output_dims->data[cls_output_dims->size - 1];
    float *classificators = cls_output_tensor->data.f;
    std::vector<float> classificatorsVec(cls_rows * cls_cols);

    // Start the camera
    cv::VideoCapture cap(0);

    // if not success, exit program
    if (cap.isOpened() == false)
    {
        std::cout << "Cannot open the camera" << std::endl;
        std::cin.get(); //wait for any key press
        return -1;
    }

    // get the frames rate of the video
    double fps = cap.get(CV_CAP_PROP_FPS);
    std::cout << "Frames per seconds : " << fps << std::endl;
    
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

    cv::String window_name = "Face Detector";
    namedWindow(window_name, cv::WINDOW_NORMAL); //create a window

    while (true)
    {
        cv::Mat rframe, frame;
        bool bSuccess = cap.read(rframe); // read a new frame from video

        // Breaking the while loop at the end of the video
        if (bSuccess == false)
        {
            std::cout << "Found the end of the video" << std::endl;
            break;
        }

        std::cout << "rframe address : " << &rframe << " frame address : " << &frame <<std::endl;

        //rframe = imread("Snap4.JPG", CV_LOAD_IMAGE_COLOR);
        int img_width = rframe.cols;
        int img_height = rframe.rows;

        // wait for for 10 ms until any key is pressed.
        // If the 'Esc' key is pressed, break the while loop.
        // If the any other key is pressed, continue the loop
        // If any key is not pressed withing 10 ms, continue the loop
        if (cv::waitKey(10) == 27)
        {
            std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
            break;
        }
            
        // OpenCV images are in BGR, model expects RGB channel format. 
        int cnls = rframe.type();
        if (cnls == CV_8UC4) 
        {
            cvtColor(rframe, frame, cv::COLOR_BGRA2RGB);
            std::cout << "This is CV_8UC4 image format" << std::endl;
        } 
        else if (cnls == CV_8UC3)
        {
            cvtColor(rframe, frame, cv::COLOR_BGR2RGB);
            std::cout << "This is CV_8UC3 image format" << std::endl;
        }
        else
        {
            std::cout << "Image format is not supported" << std::endl;
            break;
        }

        // Resize the images as required by the model.
        resize(frame, frame, cv::Size(LGT_FACE_DETECTION_MODEL_SIZE, 
                                      LGT_FACE_DETECTION_MODEL_SIZE), 0, 0, cv::INTER_LINEAR);

        // Image normalization based on std and mean (p' = (p-mean)/std)
        frame.convertTo(frame, CV_32FC3, 1 / LGT_IMAGE_NORM_STD, 
                        -LGT_IMAGE_NORM_MEAN / LGT_IMAGE_NORM_STD);

        if (!frame.isContinuous())
        {
            std::cout << "Frame NOT in Continous memory" << std::endl;
            break;
        }

        // Copy image into input tensor
        memcpy(dst, frame.data, (sizeof(float) * LGT_FACE_DETECTION_MODEL_SIZE * 
                                LGT_FACE_DETECTION_MODEL_SIZE * 3));

        // Run Inference - Interpreter invoked
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

        // copy output tensor
		memcpy(&(regressorVec[0]), regressors, sizeof(float) * reg_rows * reg_cols);
		memcpy(&(classificatorsVec[0]), classificators, sizeof(float) * cls_rows * cls_cols);

        std::vector<Detection> detections = lgt_process_cpu(regressorVec, classificatorsVec, anchors, options);
        //detections = lgt_orig_nms(detections, 0.85);
        detections = lgt_orig_nms(detections, 0.30);

        std::cout << "-----------n";
        std::cout << "detections : \n";
        std::cout << "number : " << detections.size() << std::endl;
        for (Detection points : detections)
        {
            std::cout << points.to_str();
            int x1 = int(img_width * points.xmin); 
            int x2 = int(img_width * (points.xmin + points.width));
            int y1 = int(img_height * points.ymin);
            int y2 = int(img_height * (points.ymin + points.height));
            x1 -= 30;
            y1 -= 100;
            x2 += 30;
            y2 += 50;
            std::cout << "x1: " << x1 << ", y1: " << y1 << "\nx2: " << x2 << ", y2: " << y2 << "\n";

            cv::Point pt1(x1, y1);
            cv::Point pt2(x2, y2);
            cv::rectangle(rframe, pt1, pt2, cv::Scalar(255, 0, 0), 2);
            
            cv::Mat croped_img;
            cv::Rect roi;
            roi.x = x1;
            roi.y = y1;
            roi.width = x2 - x1;
            roi.height = y2 - y1;
            std::cout << "\nimg_width: " << rframe.rows << ", img_height: " << rframe.cols << "\n";
            std::cout << "\nroi.x: " << x1 << ", roi.y: " << y1 << "\nroi.width: " << roi.width << ", roi.height: " << roi.height << "\n";

            if ((roi.x < 0) || (roi.y < 0) || (roi.width < 0) || (roi.height < 0) || 
                (img_width < (roi.x + roi.width)) || (img_height < (roi.height + roi.y)))
            {
                std::cout << "There are negative co-ordinates" << std::endl;
            }
            else
            {
                //if (img_width < (roi.x + roi.width) || img_height < (roi.height + roi.y))
                //{
                //    std::cout << "Height and Width are out of range\n";
                //}
                //else
                //{
                    croped_img = rframe(roi);
                    imshow("Cropped Image", croped_img);
                //}
            }

        }

        // show the frame in the created window
        imshow(window_name, rframe);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
