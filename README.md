# Innovative Approaches to Multi-User Tracking in Retail Spaces

This repository contains the implementation code for the research project "Innovative Approaches to Multi-User Tracking in Retail Spaces." The project focuses on developing an innovative system for continuous multi-user tracking within retail environments, emphasizing robust single-camera tracking and effective cross-camera re-identification (Re-ID).

## Abstract

This research details the development of an innovative system for continuous multi-user tracking within retail environments, focusing on robust single-camera tracking and effective cross-camera re-identification (Re-ID). A key contribution for single-camera tracking is the application of tailored post-processing heuristics addressing phantom tracks and manikin misclassifications to a state-of-the-art pipeline (YOLOv12, BoT-SORT, OSNet). The primary innovation lies in the cross-camera Re-ID methodology, which overcomes the limitations of appearance-only approaches in challenging retail settings. This is achieved through a robust spatiotemporal matching strategy utilizing homography to establish a common ground plane for coordinate-based comparisons. A novel exponential loss function is introduced for similarity scoring, significantly improving matching accuracy, particularly in scenarios with partial occlusions or non-ideal detections. The system architecture proposes a multi-stage matching process, where spatiotemporal constraints primarily guide Re-ID, with aggregated appearance features from multiple confirmed views providing a fallback mechanism, ensuring high accuracy and stability in generating comprehensive visitor trajectories.

## Methodology

### Single-Camera Multiple Object Tracking: Formalization

**Input Space**
The input to the single-camera MOT system is a sequence of video frames from a single camera, along with associated metadata (e.g., timestamps).
* **Frame Sequence**: Let $\mathcal{I} = \{I_1, I_2, \ldots, I_K\}$ denote the sequence of $K$ frames, where each frame $I_k$ at time $k$ is an image in the space $\mathcal{I}_k \subseteq \mathbb{R}^{H \times W \times C}$. Here, $H$ is the height, $W$ is the width, and $C=3$ is the number of color channels (RGB) for the 30 FPS color video.
* **Timestamps**: Each frame $I_k$ is associated with a timestamp $t_k \in \mathbb{R}_{\geq 0}$, assuming time synchronization across cameras. The sequence of timestamps is $\mathcal{T} = \{t_1, t_2, \ldots, t_K\}$, where $t_{k+1} - t_k \approx \frac{1}{30}$ seconds (for 30 FPS).
* **Input Space**: The input space is the Cartesian product of frame and timestamp sequences:
    $$\mathcal{S}_{\text{input}} = \mathcal{I} \times \mathcal{T} = \{ (I_1, t_1), (I_2, t_2), \ldots, (I_K, t_K) \}$$
    where each $(I_k, t_k) \in \mathbb{R}^{H \times W \times 3} \times \mathbb{R}_{\geq 0}$.

**State Space**
The state of an object (person) in a frame describes its properties, such as position and velocity, in the context of tracking:
* **Object State**: For the $j$-th person in frame $k$, the state is denoted $\mathbf{x}_k^j \in \mathcal{X}$, where $\mathcal{X}$ is the state space. Typically, the state includes:
    * **Position**: The bounding box center $(c_x, c_y)$ in pixel coordinates
    * **Size**: The width $w$ and height $h$ of the bounding box
    * **Velocity**: Optional components $(v_x, v_y)$ for motion prediction.
    Thus, the representation is:
    $$\mathbf{x}_k^j = [c_x, c_y, w, h, v_x, v_y]^T \in \mathbb{R}^6$$
    **Note:** in open-cv library bounding boxes described not by center coordinates + (w, h), but by coordinates of top-left corner of bounding box.
* **Frame States**: The collection of states for all $N_k$ persons in frame $k$ is:
    $$\mathbf{X}_k = (\mathbf{x}_k^1, \mathbf{x}_k^2, \ldots, \mathbf{x}_k^{N_k}) \in \mathcal{X}^{N_k}$$
* **Sequential States**: The states of the $j$-th person across frames $a$ to $b$ form a tracklet:
    $$\mathbf{x}_{a:b}^j = \{\mathbf{x}_a^j, \mathbf{x}_{a+1}^j, \ldots, \mathbf{x}_b^j\} \in \mathcal{X}^{b-a+1}$$

**Measurement Space**
In the Tracking-by-Detection paradigm, measurements are derived from a detector (YOLO in my case). For the $j$-th person in frame $k$, the measurement $\mathbf{z}_k^j \in \mathcal{Z}$ typically includes:
    $$\mathbf{z}_k^j = [z_x, z_y, z_w, z_h, s]^T \in \mathbb{R}^5$$
    where $(z_x, z_y)$ is the bounding box center, $(z_w, z_h)$ is the width and height, and $s \in [0,1]$ is the detection confidence. All measurements in frame $k$ are:
    $$\mathbf{Z}_k = (\mathbf{z}_k^1, \mathbf{z}_k^2, \ldots, \mathbf{z}_k^{M_k}) \in \mathcal{Z}^{M_k}$$

**Output Space (Tracklets)**
The output of the single-camera MOT system is a set of tracklets $\mathcal{T} = \{ T^1, T^2, \ldots, T^J \}$, where a tracklet for the $j$-th person is $T^j = (id_j, \mathbf{x}_{a:b}^j)$ where $id_j \in \mathbb{N}$ is a unique identifier.

**Appearance Features (ReID)**
For each detection $\mathbf{z}_k^j$, the ReID model extracts:
    $$\mathbf{f}_k^j \in \mathcal{F} = \mathbb{R}^d$$
    for further inter-camera appearance-based ReID.

**Objective**
The objective is to estimate the optimal sequential states for all persons, formulated as a Maximum a Posteriori (MAP) estimation problem:
$$\mathbf{X}_{1:k}^* = \arg\max_{\mathbf{X}_{1:k}} p(\mathbf{X}_{1:k} | \mathbf{Z}_{1:k})$$
This seeks the most probable set of trajectories given all observed measurements.

### Cross-Camera Person Re-identification: Formalization

**Input Space**
The input to the Cross-Camera ReID system consists of tracklets from multiple cameras, generated by the single-camera MOT pipeline (as formalized previously), along with camera metadata (e.g., timestamps and homography mappings).
* **Cameras**: Let $\mathcal{C} = \{C_1, C_2, \ldots, C_M\}$ denote the set of $M$ cameras in the mall.
* **Tracklets per Camera**: For camera $C_m$, the single-camera MOT system produces a set of tracklets:
    $$\mathcal{T}_m = \{ T_m^1, T_m^2, \ldots, T_m^{J_m} \}$$
    where $T_m^j = (id_m^j, \mathbf{x}_{a:b}^{m,j})$ is the tracklet for the $j$-th person in camera $C_m$, with local ID $id_m^j \in \mathbb{N}$ and state sequence $\mathbf{x}_{a:b}^{m,j} = \{\mathbf{x}_a^{m,j}, \ldots, \mathbf{x}_b^{m,j}\} \in \mathcal{X}^{b-a+1}$. The state $\mathbf{x}_k^{m,j} \in \mathcal{X} = \mathbb{R}^6$ includes position, size, and velocity (e.g., $\mathbf{x}_k^{m,j} = [c_x, c_y, w, h, v_x, v_y]^T$).
* **Appearance Features**: Each tracklet $T_m^j$ is associated with a sequence of appearance features from the ReID model (e.g., OSNet):
    $$\mathbf{f}_{a:b}^{m,j} = \{\mathbf{f}_a^{m,j}, \ldots, \mathbf{f}_b^{m,j}\} \in \mathcal{F}^{b-a+1}$$
    where $\mathbf{f}_k^{m,j} \in \mathcal{F} = \mathbb{R}^d$ (e.g., $d=512$ for OSNet) is the feature embedding for the $j$-th person in frame $k$ of camera $C_m$.
* **Timestamps**: Each state $\mathbf{x}_k^{m,j}$ is associated with a timestamp $t_k^m \in \mathbb{R}_{\geq 0}$, synchronized across cameras (i.e., $t_k^m \approx t_k^n$ for cameras $C_m$ and $C_n$ at frame $k$).
* **Homography for Overlapping Cameras**: For a pair of cameras $(C_m, C_n)$ with overlapping fields of view, a homography matrix $H_{m,n} \in \mathbb{R}^{3 \times 3}$ maps pixel coordinates from camera $C_m$ to camera $C_n$. Let $\mathcal{H} = \{ H_{m,n} \mid (C_m, C_n) \text{ have overlapping views} \}$ denote the set of homography matrices.

Overall, the input space combines tracklets, appearance features, timestamps, and homography mappings:
$$\mathcal{S}_{\text{input}} = \prod_{m=1}^M \mathcal{T}_m \times \prod_{m=1}^M \mathcal{F}_m \times \mathcal{T}_{\text{time}} \times \mathcal{H}$$
where $\mathcal{F}_m = \{\mathbf{f}_{a:b}^{m,j} \mid T_m^j \in \mathcal{T}_m\}$ is the set of feature sequences for camera $C_m$, and $\mathcal{T}_{\text{time}} = \prod_{m=1}^M \{t_k^m \mid k=1, \ldots, K_m\}$ is the set of timestamps across all cameras.

**Output Space**
The output of the Cross-Camera ReID system is a set of global tracklets, where tracklets from different cameras representing the same person are merged in one set and assigned a shared global ID. A global tracklet for the $j$-th person is:
$$T^j_{\text{global}} = (id_{\text{global}}^j, \{ T_{m_1}^{j_1}, T_{m_2}^{j_2}, \ldots, T_{m_p}^{j_p} \})$$
where $id_{\text{global}}^j \in \mathbb{N}$ is a unique global ID, and $\{ T_{m_1}^{j_1}, \ldots, T_{m_p}^{j_p} \}$ is the set of camera-specific tracklets (from cameras $C_{m_1}, \ldots, C_{m_p}$) corresponding to the same person. The set of all global tracklets is:
$$\mathcal{T}_{\text{global}} = \{ T^1_{\text{global}}, T^2_{\text{global}}, \ldots, T^J_{\text{global}} \}$$
where $J$ is the total number of unique persons in the mall. The output space is:
$$\mathcal{S}_{\text{output}} = 2^{\mathbb{N} \times \bigcup_{m_1, \ldots, m_p} (\mathcal{T}_{m_1} \times \cdots \times \mathcal{T}_{m_p})}$$
(the power set of global IDs paired with combinations of camera-specific tracklets).

**Spatiotemporal Alignment (Homography-Based)**
* **Coordinate Mapping.** For a tracklet state $\mathbf{x}_k^{m,j} = [c_x, c_y, w, h, v_x, v_y]^T$ in camera $C_m$, the position $(c_x, c_y)$ is mapped to camera $C_n$ using homography $H_{m,n}$:
    $$\begin{bmatrix} c_x' \\ c_y' \\ 1 \end{bmatrix} = H_{m,n} \begin{bmatrix} c_x \\ c_y \\ 1 \end{bmatrix}$$
    where $(c_x', c_y')$ are the homogeneous coordinates in camera $C_n$, normalized to Cartesian coordinates by dividing by the third component.
* **Shared Coordinate Space.** Assume a common physical plane (here the mall floor). For each camera $C_m$, a homography $H_m$ maps pixel coordinates to a global coordinate system $\mathcal{G} = \mathbb{R}^2$. For a state $\mathbf{x}_k^{m,j}$, the global position is:
    $$\mathbf{g}_k^{m,j} = H_m ([c_x, c_y, 1]^T) \in \mathcal{G}$$
* **Spatiotemporal Constraint.** Two tracklets $T_m^j = (id_m^j, \mathbf{x}_{a:b}^{m,j})$ and $T_n^i = (id_n^i, \mathbf{x}_{c:d}^{n,i})$ are candidates for the same person if, for some frame $k$ in their overlapping time interval $[t_{\max(a,c)}, t_{\min(b,d)}]$, their global positions are close:
    $$\| \mathbf{g}_k^{m,j} - \mathbf{g}_k^{n,i} \|_2 < \epsilon$$
    where $\epsilon$ is a spatial tolerance (e.g., 0.5 meters), and timestamps $t_k^m \approx t_k^n$ (due to synchronization) *Details of matching algorithm explained in corresponding section*.

**Appearance-Based Matching.**
For non-overlapping cameras or to refine spatiotemporal matches, appearance features are used:
* **Feature Distance**: For tracklets $T_m^j$ and $T_n^i$, compute a similarity score (e.g., cosine similarity) between their feature sequences:
    $$s(\mathbf{f}_{a:b}^{m,j}, \mathbf{f}_{c:d}^{n,i}) = \frac{1}{L} \sum_{k \in \text{overlap}} \frac{\mathbf{f}_k^{m,j} \cdot \mathbf{f}_k^{n,i}}{\|\mathbf{f}_k^{m,j}\|_2 \|\mathbf{f}_k^{n,i}\|_2}$$
    where $L$ is the number of overlapping frames, and $s \in [-1, 1]$. A threshold $\theta_{\text{appearance}}$ (e.g., 0.8) determines a match.

**Objective.**
The Cross-Camera ReID problem is to assign global IDs to tracklets by solving a clustering problem:
$$\mathcal{T}_{\text{global}}^* = \arg\max_{\mathcal{T}_{\text{global}}} p(\mathcal{T}_{\text{global}} | \mathcal{T}_1, \ldots, \mathcal{T}_M, \mathcal{H}, \{\mathbf{f}_{a:b}^{m,j}\})$$
This can be formulated as minimizing an energy function:
$$\mathcal{T}_{\text{global}}^* = \arg\min_{\mathcal{T}_{\text{global}}} E(\mathcal{T}_{\text{global}} | \mathcal{T}_1, \ldots, \mathcal{T}_M, \mathcal{H}, \{\mathbf{f}_{a:b}^{m,j}\})$$
where the energy function $E$ includes:
* **Spatiotemporal Term**: Penalizes mismatches in global coordinates and timestamps for overlapping cameras.
    $$E_{\text{st}} = \sum_{(m,n) \in \mathcal{H}} \sum_{k} \sum_{(j,i)} \mathbb{1}_{t_k^m \approx t_k^n} \cdot \max(0, \| \mathbf{g}_k^{m,j} - \mathbf{g}_k^{n,i} \|_2 - \epsilon)$$
* **Appearance Term**: Penalizes dissimilar features for tracklet pairs assigned the same global ID.
    $$E_{\text{app}} = \sum_{T^j_{\text{global}}} \sum_{(T_m^a, T_n^b) \in T^j_{\text{global}}} (1 - s(\mathbf{f}_{a:b}^{m,a}, \mathbf{f}_{c:d}^{n,b}))$$
* **Consistency Term**: Ensures each tracklet belongs to exactly one global tracklet.

**Optimization Framework**
The energy minimization problem can be solved using a **graph-based clustering approach** (e.g. K-Means):
* **Nodes**: Camera-specific tracklets $T_m^j$.
* **Edges**: Connect tracklets with high spatiotemporal or appearance similarity.
* **Objective**: Partition the graph into clusters, each representing a global tracklet $T^j_{\text{global}}$, minimizing $E$.

Alternatively, a probabilistic approach can model the joint probability:
$$p(\mathcal{T}_{\text{global}} | \mathcal{T}_1, \ldots, \mathcal{T}_M, \mathcal{H}, \{\mathbf{f}_{a:b}^{m,j}\}) \propto \prod_{T^j_{\text{global}}} p_{\text{st}}(T^j_{\text{global}} | \mathcal{H}) \cdot p_{\text{app}}(T^j_{\text{global}} | \{\mathbf{f}_{a:b}^{m,j}\})$$
where $p_{\text{st}}$ and $p_{\text{app}}$ are likelihoods for spatiotemporal and appearance consistency, respectively.

**Mapping to Global Tracklets**
The ReID system maps the input $\mathcal{S}_{\text{input}}$ to the output $\mathcal{S}_{\text{output}}$ via a function:
$$\Psi: \mathcal{S}_{\text{input}} \to \mathcal{S}_{\text{output}}$$
where $\Psi(\{\mathcal{T}_m\}, \{\mathbf{f}_{a:b}^{m,j}\}, \mathcal{T}_{\text{time}}, \mathcal{H}) = \mathcal{T}_{\text{global}}$.

### Methodology: Single-Camera Multiple Object Tracking.

**Components**
Since the Tracking by Detection (TBD) approach prove it’s effectiveness (see Table 1) and it's well-integrated, I choose this paradigm. Offline methods excels in accuracy, they was too costly in the computational sense for me. Therefore I was pushed to use **online** methods.

As a detection model I choose **YOLOv12** since it’s best SOTA solution available according to COCO benchmark (see Table 1).

As a tracker I’ve tried BotSORT, ByteTrack and DeepSORT. According to my literature review block and MOT20 benchmark, BotSORT and ByteTrack excels old DeepSORT (despite I appreciate it’s integration). At the end I decide to choose BotSORT, because it excels in crowded-dense scenarios. And as the inter-camera ReID model for ByteTrack (adapted for BotSORT) I choose OSNet model. According to my literature review block Omni-Scale Network is overperform most of ReID inter-camera models (at least at its size, because inference of CLIP models was painful. Also it's well-integrated in TorchREID library, which significantly simplified my life.

This pipeline considered is SOTA in 2025, however doesn’t bring novelty. Therefore let’s move to post-processing heuristics.

**Post-processing ReID to Mitigate Identity Switches**
Despite the integration of a robust ReID model (OSNet) within our single-camera MOT pipeline (YOLOv12, BoT-SORT), the nature of online, detection-based tracking can still lead to identity switching, particularly when individuals remain occluded for extended periods (e.g., more than 2 seconds). When an object reappears after such an occlusion, the tracker might assign it a new ID, fragmenting its trajectory. To address this, we implement a post-processing ReID step aimed at re-associating these fragmented tracklets.

The fundamental approach for this intra-camera ReID post-processing is analogous to the appearance-only strategy detailed for cross-camera ReID (see Section 7.1: Attempts of Using Appearance-Only Approach), adapted here for re-linking tracklets within the same camera view. The core idea is to use a query-gallery framework based on visual appearance features.

* **Methodology**
    When a tracklet, say $T_{A}$, terminates (e.g., due to occlusion or temporarily leaving the frame) and shortly thereafter a new tracklet, $T_{B}$, initiates in a spatially proximate region, $T_{A}$ becomes a "query" and $T_{B}$ (along with other newly initiated tracklets in the vicinity) forms a "gallery" of candidates for re-association.
    1.  **Feature Extraction and Aggregation:**
        * For the query tracklet $T_{A}=(id_{A}, x_{a:b}^{A}, f_{a:b}^{A})$, which ended at frame $b$, we consider its appearance features $f_{k}^{A}$ from its last few observed frames. These are aggregated into a representative feature vector $\overline{f}^{A}$ using averaging, similar to Equation 4:
            $$\overline{f}^{A} = \frac{1}{N_A} \sum_{k=b-N_A+1}^{b} f_{k}^{A}$$
            where $N_A$ is the number of recent frames from $T_A$ used for representation.
        * For each candidate tracklet $T_{B}=(id_{B}, x_{c:d}^{B}, f_{c:d}^{B})$ in the gallery, which started at frame $c$, we use its initial appearance features $f_{k}^{B}$. These are similarly aggregated into $\overline{f}^{B}$:
            $$\overline{f}^{B} = \frac{1}{N_B} \sum_{k=c}^{c+N_B-1} f_{k}^{B}$$
            where $N_B$ is the number of initial frames from $T_B$ used.
    2.  **Similarity Scoring:**
        The similarity between the query tracklet $T_A$ and a candidate tracklet $T_B$ is computed using the cosine similarity between their aggregated feature vectors:
        $$s(\overline{f}^{A}, \overline{f}^{B}) = \frac{\overline{f}^{A} \cdot \overline{f}^{B}}{\|\overline{f}^{A}\|_{2} \|\overline{f}^{B}\|_{2}}$$
        This score $s \in [-1, 1]$ indicates the visual similarity, with higher values suggesting a stronger match.
    3.  **Matching and Merging:**
        A match is confirmed if the similarity score $s(\overline{f}^{A}, \overline{f}^{B})$ exceeds a predefined threshold, $\theta_{\text{relink}}$ (e.g., $\theta_{\text{relink}} = 0.85$, determined empirically). If a match is found with the highest similarity score above this threshold, the tracklets $T_A$ and $T_B$ are considered to represent the same individual. Consequently, tracklet $T_B$ is merged into $T_A$ by assigning $id_A$ to all observations in $T_B$ and concatenating their state and feature sequences. A greedy approach is used to assign matches, ensuring each new tracklet is linked to at most one prior tracklet in this post-processing phase.
* **Justification and Effectiveness**
    This appearance-based re-linking strategy is adopted due to its demonstrated effectiveness in challenging ReID scenarios and its relative simplicity for this post-processing stage. The underlying assumption is that even if the online tracker loses an identity due to prolonged occlusion, the appearance features extracted upon re-detection should be sufficiently similar to those captured before occlusion. The effectiveness of the learned appearance features (from OSNet) in distinguishing identities is illustrated in Figure 1. The t-SNE visualization shows that feature embeddings for the same individual (represented by a unique color) tend to form distinct clusters, separated from the clusters of other individuals. This clear separation in the feature space supports the viability of using appearance similarity to correctly re-cluster or re-link track fragments that were erroneously assigned different IDs by the online tracker.
    For now, this basic visual ReID approach is employed without additional complex heuristics for the intra-camera ID switch correction, relying on the strength of the feature embeddings. Future work could explore incorporating motion continuity or short-term spatio-temporal predictions as additional cues to further refine the re-linking decisions.


## Code Description

This repository includes the following key Python scripts:

* `dp_byte_fast.py`: Implements object tracking using YOLO for detection and ByteTrack for tracking. It processes an input video, performs detections, updates the tracker, annotates frames, and saves the output video and tracklet data.
* `dst_YOLO.py`: Implements object tracking using YOLO for detection and DeepSort for tracking. It processes video, performs detections, converts them for DeepSort, updates the tracker, saves tracklets, and outputs an annotated video.
* `link_gallery_to_prev.py`: Links identities across different galleries (e.g., from different camera views or time segments) using ReID models (traditional or CLIP-based). It extracts features, finds best matches using a voting mechanism, and can apply these links to remap tracklet IDs.
* `match_tracklets_coords.py`: Matches tracklets based on real-world coordinates and timestamps. It defines a loss function considering time and distance differences, processes query tracklets against base tracklets to find the best match using a weighted voting system, and can visualize these matches.
* `reid_on_gallery.py`: Performs ReID within a single gallery of tracklet features. It computes similarities between track features, finds matches above a threshold, and clusters identities to merge tracks belonging to the same person.
* `reid_on_gallery_pictures.py`: Processes a gallery of cropped images (organized by track ID) using ReID models (CLIP) to merge similar identities. It extracts features from images, compares them to an existing ReID database or builds one, maps original track IDs to merged ReID IDs, and can apply this mapping to tracklet files.
* `rewrite_tracklets.py`: A utility script for post-processing tracklet files. It can remap tracklet IDs based on clustering results, filter out short tracks, remove tracks with a high percentage of zero-confidence detections, and filter out stationary objects (manikins).

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  Install dependencies. It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    (Note: You'll need to create a `requirements.txt` file listing all necessary packages like `opencv-python`, `torch`, `torchvision`, `ultralytics`, `numpy`, `ByteTrack` (ensure ByteTrack is installable via pip or provide setup instructions), `deep_sort_realtime`, `torchreid`, `openai-clip` etc.)

## Usage

Each script can be run from the command line. Refer to the arguments or configuration sections within each script for specific parameters like input video paths, model paths, output directories, etc.

Example (generic):
```bash
python dp_byte_fast.py --input video.mp4 --model yolov12.pt --output_dir ./outputs
```