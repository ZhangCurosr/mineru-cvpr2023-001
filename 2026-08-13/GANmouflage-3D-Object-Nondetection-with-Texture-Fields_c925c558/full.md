![](images/31bb830a8e233b9615529648342e8f2bd4693a73d1b63864e81222e263126782.jpg)

# GANmouflage: 3D Object Nondetection with Texture Fields

Rui Guo<sup>1,3\*</sup> Jasmine Collins<sup>2</sup> University of Michigan<sup>1</sup>

Oscar de Lima<sup>1</sup> UC Berkeley<sup>2</sup>

Andrew Owens<sup>1</sup> XMotors.ai<sup>3</sup>

![](images/ee9056e20fd3d057f8981b1e9a9c810fc0e6272a8eb314956103f178707876b3.jpg)

![](images/4fe18c69e690e6a244e6474ce7fa11289880d3abd74dbff1d79ce9c317f0df53.jpg)

![](images/5689d19034ee91953f9eae7e6e21d542b40bd47cdd66c63b4aaec2695937d870.jpg)

![](images/015123ce1fe33550ad23de71237625c50020c6b0896d7911e98b9816b738fb8c.jpg)  
(a) Camou aged object

![](images/cb1711a097fcfad1a7390cd731bd44a92d5c85549cb333868f2a1f231f857226.jpg)  
(b) Viewpoints of camou aged object

![](images/35135d6e471481dbf5bc9a22f1cd5a6865a1445e03459973413604adce2709be.jpg)  
(c) Object without camou age

flFigure 1. We learn to camouflage 3D objects within scenes. Given an object’s shape, position, and a distribution of possible viewpoints it will be seen from, we estimate a texture field that will conceal it. We show example outputs from our model, with two viewpoints for each camouflaged object. Please see the videos on our webpage (https://rrrrrguo.github.io/ganmouflage) for more examples.

## Abstract

We propose a method that learns to camouflage 3D objects within scenes. Given an object’s shape and a distribution of viewpoints from which it will be seen, we estimate a texture that will make it difficult to detect. Successfully solving this task requires a model that can accurately reproduce textures from the scene, while simultaneously dealing with the highly conflicting constraints imposed by each viewpoint. We address these challenges with a model based on texture fields and adversarial learning. Our model learns to camouflage a variety of object shapes from randomly sampled locations and viewpoints within the input scene, and is the first to address the problem of hiding complex object shapes. Using a human visual search study, wefind that our estimated textures conceal objects significantly better than previous methods.

## 1. Introduction

Using fur, feathers, spots, and stripes, camouflaged animals show a remarkable ability to stay hidden within their environment. These capabilities developed as part of an evolutionary arms race, with advances in camouflage leading to advances in visual perception, and vice versa.

Inspired by these challenges, previous work [33] proposed the object nondetection problem: to create an appearance for an object that makes it undetectable. Given an object’s shape and a sample of photos from a scene, the goal is to produce a texture that hides the object from every viewpoint that it is likely to be observed from. This problem has applications in hiding unsightly objects, such as utility boxes [7], solar panels [29, 49], and radio towers, and in concealing objects from humans or animals, such as surveillance cameras and hunting platforms. Moreover, since camouflage models must ultimately thwart highly effective visual systems, they may provide a better scientific understanding of the cues that these systems use. Animal camouflage, for instance, has developed strategies for avoiding perceptual grouping and boundary detection cues [30, 52].

A successful learning-based camouflage system, likewise, must gain an understanding of these cues in order to thwart them.

Previous object nondetection methods are based on nonparametric texture synthesis. Although these methods have shown success in hiding cube-shaped objects, they can only directly “copy-and-paste” pixels that are directly occluded by the object, making it challenging to deal with complex backgrounds and non-planar geometry. While learningbased methods have the potential to address these shortcomings, they face a number of challenges. Since even tiny imperfections in synthesized textures can expose a hidden object, the method must also be capable of reproducing realworld textures with high fidelity. There is also no single texture that can perfectly conceal an object from all viewpoints at once. Choosing an effective camouflage requires 3D reasoning, and making trade-offs between different solutions. This is in contrast to the related problem of image inpainting, which can be posed straightforwardly as estimating masked image regions in large, unlabeled photo collections [34], and which lack the ability to deal with multiview constraints.

We propose a model based on neural texture fields [23, 32,35,42] and adversarial training that addresses these challenges (Figure 2). The proposed architecture and learning procedure allow the model to exploit multi-view geometry, reproduce a scene’s textures with high fidelity, and satisfy the highly conflicting constraints provided by the input images. During training, our model learns to conceal a variety of object shapes from randomly chosen 3D positions within a scene. It uses a conditional generative adversarial network (GAN) to learn to produce textures that are difficult to detect using pixel-aligned representations [55] with hypercolumns [20] to provide information from each view.

Through automated evaluation metrics and human perceptual studies, we find that our method significantly outperforms the previous state-of-the-art in hiding cuboid objects. We also demonstrate our method’s flexibility by using it to camouflage a diverse set of complex shapes. These shapes introduce unique challenges, as each viewpoint observes a different set of points on the object surface. Finally, we show through ablations that the design of our texture model leads to significantly better results.

## 2. Related Work

Computational camouflage We take inspiration from early work by Reynolds [38] that formulated camouflage as part of an artificial life simulation, following Sims [45] and Dawkins [13]. In that work, a human “predator” interactively detects visual “prey” patterns that are generated using a genetic algorithm. While our model is also trained adversarially, we do so using a GAN, rather than with a human-in-the-loop. Later, Owens et al. [33] proposed the problem of hiding a cuboid object at a specific location from multiple 3D viewpoints, and solved it using nonparametric texture synthesis. In contrast, our model learns through adversarial training to hide both cuboid and more complex objects. Bi et al. [5] proposed a patch-based synthesis method that they applied to the multi-view camouflage problem, and extended the method to spheres. However, this work was very preliminary: they only provide a qualitative result on a single scene (with no quantitative evaluation). Other work inserts difficult-to-see patterns into other images [10, 58].

Animal camouflage. Perhaps the most well-known camouflage strategy is background matching, whereby animals take on textures that blend into the background. However, animals also use a number of other strategies to conceal themselves, such as by masquerading as other objects [48], and using disruptive coloration to elude segmentation cues and to hide conspicuous body parts, such as eyes [12]. The object nondetection problem is motivated by animals that can dynamically change their appearance to match their surroundings, such as the octopus<sup>1</sup> [19]. Researchers have also begun using computational models to study animal camouflage. Troscianko et al. [53] used a genetic algorithm to camouflage synthetic bird eggs, and asked human subjects to detect them. Talas et al. [52] used a GAN to camouflage simple triangle-shaped representations of moths that were placed at random locations on synthetic tree bark. In both cases, the animal models are simplified and 2D, whereas our approach can handle complex 3D shapes.

Camouflaged object detection. Recent work has sought to detect camouflaged objects using object detectors [15, 28, 54] and motion cues [8, 27]. The focus of our work is generating camouflaged objects, rather than detecting them. Adversarial examples. The object nondetection problem is related to adversarial examples [6, 18, 51], in that both problems involve deceiving a visual system (e.g., by concealing an object or making it appear to be from a different class). Other work has generalized these examples to multiple viewpoints [2]. In contrast, the goal of the nondetection problem is to make objects that are concealed from a human visual system, rather than fool a classifier.

Texture fields. We take inspiration from recent work that uses implicit representations of functions to model the surface texture of objects [23, 32, 35, 42]. Oechsle et al. [32] learned to texture a given object using an implicit function, with image and shape encoders, and Saito et al. [42] learned a pixel-aligned implicit function for clothed humans. There are three key differences between our work and these methods. First, these methods aim to reconstruct textures from given images while our model predicts a texture that can conceal an object. Second, our model is conditioned on a 3D input scene with projective structure, rather than a set of images. Finally, the constraints provided by our images are mutually incompatible: there is no single way to texture a 3D object that satisfies all of the images. Other work has used implicit functions to represent 3D scenes for view synthesis [9, 31, 46, 55]. Sitzmann et al. [46] proposed an implicit 3D scene representation. Mildenhall et al. [31] proposed view-dependent neural radiance fields (NeRF). Recent work created image-conditional NeRFs [9, 55]. Like our method, they use networks with skip connections that exploit the projective geometry of the scene. However, their learned radiance field does not ensure multi-view consistency in color, since colors are conditioned on viewing directions of novel views.

![](images/625f3c780d2830fb2215322ae6b9d73a7d13fe3f67889aa99a84098e3e01ae59.jpg)  
(a) Multi-view camouflage

![](images/2f5603ec2c0caeba360f4ef5d1227429b8dcc131f60fc681e798493643e646ae.jpg)  
(b) Texture model

![](images/285b4e7b2564d79b5914da0a15a3e466088c8f90b92d74b411b4ac86b61a5f27.jpg)  
(d) Adversarial loss  
Figure 2. Camouflage model. (a) Our model creates a texture for a 3D object that conceals it from multiple viewpoints. (b) We generate a texture field that maps 3D points to colors. The network is conditioned on pixel-aligned features from training images. We train the model to create a texture that is (c) photoconsistent with the input views, as measured using a perceptual loss, and (d) difficult for a discriminator to distinguish from random background patches. For clarity, we show the camouflaged object’s boundaries.

Inpainting and texture synthesis. The camouflage problem is related to image inpainting [3, 4, 14, 21, 34, 57], in that both tasks involve creating a texture that matches a surrounding region. However, in contrast to the inpainting problem, there is no single solution that can completely satisfy the constraints provided by all of the images, and thus the task cannot be straightforwardly posed as a selfsupervised data recovery problem [34]. Our work is also related to image-based texture synthesis [3, 14, 17] and 3D texture synthesis [23, 32, 35]. Since these techniques fill a hole in a single image, and cannot obtain geometricallyconsistent constraints from multiple images, they cannot be applied to our method without major modifications. Nevertheless, we include an inpainting-based baseline in our evaluation by combining these methods with previous camouflage approaches.

## 3. Learning Multi-View Camouflage

Our goal is to create a texture for an object that camouflages it from all of the viewpoints that it is likely to be observed from. Following the formulation of Owens et al. [33], our input is a 3D object mesh at a fixed location in a scene, a sample of photos $I _ { 1 } , I _ { 2 } , . . . , I _ { N }$ from distribution , and their camera parameters ${ \bf K } _ { j } , { \bf R } _ { j } , { \bf t } _ { j }$ . We desire a solution that camouflages the object from , using this sample. We are also provided with a ground plane g, which the object has been placed on.

Also following [33], we consider the camouflage problem separately from the display problem of creating a realworld object. We assume that the object can be assigned arbitrary textures, and that there is only a single illumination condition. We note that shadows are independent of the object texture, and hence could be incorporated into this problem framework by inserting shadows into images (Sec. 4.5). Moreover, changes in the amount of lighting are likely to affect the object and background in a consistent way, producing a similar camouflage.

## 3.1. Texture Representation

We create a surface texture for the object that, on average, is difficult to detect when observed from viewpoints randomly sampled from . As in prior work [33], we render the object and synthetically insert it into the scene.

Similar to recent work on object texture synthesis [23, 32, 35], we represent our texture as continuous function in 3D space, using a texturefield:

$$
t _ { \theta } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 } .\tag{1}
$$

This function maps a 3D point to an RGB color, and is parameterized using a multi-layer perceptron (MLP) with weights θ.

We condition our neural texture representation on input images, their projection matrices ${ \bf P } _ { j } = { \bf K } _ { j } [ { \bf R } _ { j } | { \bf t } _ { j } ]$ , and a 3D object shape . Our goal is to learn a texturingfunction that produces a texture field from an input scene:

$$
G _ { \theta } ( \mathbf { x } ; \{ \mathbf { I } _ { j } \} , \{ \mathbf { P } _ { j } \} , S )\tag{2}
$$

where x is a 3D query point on the object surface.

## 3.2. Camouflage Texture Model

To learn a camouflaged texture field (Eq. 2), we require a representation for the multi-view scene content, geometry, and texture field. We now describe these components in more detail. Our full model is shown in Figure 2.

Pixel-aligned image representation. In order to successfully hide an object, we need to reproduce the input image textures with high fidelity. For a given 3D point $\mathbf { x } _ { i }$ on the object surface and an image $\mathbf { I } _ { j }$ , we compute an image feature $\mathbf { z } _ { i } ^ { ( j ) }$ as follows.

We first compute convolutional features for $\mathbf { I } _ { j }$ using a U-net [40] with a ResNet-18 [22] backbone at multiple resolutions. We extract image features $\mathbf { F } ^ { ( j ) } = E ( \mathbf { I } _ { j } )$ at full, $\textstyle { \frac { 1 } { 4 } } .$ , and $\frac { 1 } { 1 6 }$ scales. At each pixel, we concatenate features for each scale together, producing a multiscale hypercolumn representation [20].

Instead of using a single feature vector to represent an entire input image, as is often done in neural texture models that create a texture from images [23, 32], we exploit the geometric structure of the multi-view camouflage problem. We extract pixel-aligned features $\mathbf { z } _ { i } ^ { ( j ) }$ from each feature map $\mathbf { F } ^ { ( j ) }$ , following work in neural radiance fields [55]. We compute the projection of a 3D point $\mathbf { x } _ { i }$ in viewpoint $\mathbf { I } _ { j } { \boldsymbol { : } }$

$$
\mathbf { u } _ { i } ^ { ( j ) } = \pi ^ { ( j ) } ( \mathbf { x } _ { i } ) ,\tag{3}
$$

where $\pi$ is the projection function from object space to screen space of image $\mathbf { I } _ { j }$ . We then use bilinear interpolation to extract the feature vector $\mathbf { z } _ { i } ^ { ( j ) } = \mathbf { F } ^ { ( j ) } ( \mathbf { u } _ { i } ^ { ( j ) } )$ for each point i in each input image $\mathbf { I } _ { j }$

Perspective encoding. In addition to the image representation, we also condition our texture field on a perspective encoding that conveys the local geometry of the object surface and the multi-view setting. For each point $\mathbf { x } _ { i }$ and image $\mathbf { I } _ { j }$ , we provide the network with the viewing direction $\mathbf { v } _ { i } ^ { ( j ) }$ and surface normal $\mathbf { n } _ { i } ^ { ( j ) }$ . These can be computed as: $\begin{array} { r } { \mathbf { v } _ { i } ^ { ( j ) } = \frac { \mathbf { K } _ { j } ^ { - 1 } \mathbf { u } _ { i } ^ { ( j ) } } { \| \mathbf { K } _ { i } ^ { - 1 } \mathbf { u } _ { i } ^ { ( j ) } \| _ { 2 } } } \end{array}$ and $\mathbf { n } _ { i } ^ { ( j ) } = \mathbf { R } _ { j } \mathbf { n } _ { i }$ , where $\mathbf { u } _ { i } ^ { ( j ) }$ is the point’s projection (Eq. 3) in homogeneous coordinates, and ${ \bf n } _ { i }$ is the surface normal in object space. To obtain $\mathbf { n } _ { i } .$ , we extract the normal of the face closet to $\mathbf { x } _ { i }$

We note that these perspective features come from the images that are used as input images to the texture field, rather than the camera viewing the texture, $i . e .$ in contrast to neural scene representations [9, 31, 55], our textures are not viewpoint-dependent.

Texture field architecture. We use these features to define a texture field, an MLP that maps a 3D coordinate $\mathbf { x } _ { i }$ to a color $\mathbf { c } _ { i } \left( \mathrm { E q . ~ 1 } \right)$ . It is conditioned on the set of image features for the N input images $\{ \mathbf { z } _ { i } ^ { ( j ) } \}$ , as well as the sets of perspective features $\{ \mathbf { v } _ { i } ^ { ( j ) } \}$ and $\{ \mathbf { n } _ { i } ^ { ( j ) } \}$

$$
\mathbf { c } _ { i } = T ( \gamma ( \mathbf { \gamma x } _ { i } ) ; \{ \mathbf { z } _ { i } ^ { ( j ) } \} , \{ \mathbf { v } _ { i } ^ { ( j ) } \} , \{ \mathbf { n } _ { i } ^ { ( j ) } \} )\tag{4}
$$

where $\gamma ( \cdot )$ is a positional encoding [31]. For this MLP, we use a similar architecture as Yu et al. [55]. The network is composed of several fully connected residual blocks and has two stages. In the first stage, which consists of 3 blocks, the vector from each input view is processed separately with shared weights. Mean pooling is then applied to create a unified representations from the views. In the second stage, another 3 residual blocks are used to predict the color for the input query point. Please see the supplementary material for more details.

Rendering. To render the object from a given viewpoint, following the strategy of Oechsle et al. [32], we determine which surface points are visible using the object’s depth map, which we compute using PyTorch3D [37]. Given a pixel $\mathbf { u } _ { i } .$ , we estimate a 3D surface point $\mathbf { x } _ { i }$ in object space through inverse projection: $\mathbf { x } _ { i } \ = \ d _ { i } \mathbf { R } ^ { T } \mathbf { K } ^ { - 1 } \mathbf { u } _ { i } \ - \ \mathbf { R } ^ { T } \mathbf { t }$ where $d _ { i }$ is the depth of pixel i, K, R, t are the view’s camera parameters, and $\mathbf { u } _ { i }$ is in homogeneous coordinates. We estimate the color for all visible points, and render the object by inserting the estimated pixel colors into a background image, I. This results in a new image that contains the camouflaged object, ˆI.

## 3.3. Learning to Camouflage

We require our camouflage model to generate textures that are photoconsistent with the input images, and that are not easily detectable by a learned discriminator. These two criteria lead us to define a loss function consisting of a photoconsistency term and adversarial loss term, which we optimize through a learning regime that learns to camouflage randomly augmented objects from random positions.

Photoconsistency. The photoconsistency loss measures how well the textured object, when projected into the input views, matches the background. We use a perceptual loss, $\mathcal { L } _ { p h o t o } \left[ 1 7 , 2 5 \right]$ that is computed as the normalized distance between activations for layers of a VGG-16 network [44] trained on ImageNet [41]:

$$
\mathcal { L } _ { p h o t o } = \sum _ { j \in J } \mathcal { L } _ { P } ( \hat { \mathbf { I } } _ { j } , \mathbf { I } _ { j } ) = \sum _ { j \in J , k \in L } \frac { 1 } { N _ { k } } \| \phi _ { k } ( \hat { \mathbf { I } } _ { j } ) - \phi _ { k } ( \mathbf { I } _ { j } ) \| _ { 1 }\tag{5}
$$

where J is the set of view indices, L is the set of layers used in the loss, and $\phi _ { k }$ are the activations of layer $k ,$ which has total dimension $N _ { k }$ . In practice, due to the large image size relative to the object, we use a crop centered around the object, rather than $\mathbf { I _ { j } }$ itself (see Figure 2(c)).

Adversarial loss. To further improve the quality of generated textures, we also use an adversarial loss. Our model tries to hide the object, while a discriminator attempts to detect it from the scene. We randomly select real image crops y from each background image $\mathbf { I } _ { j }$ and select fake crops $\hat { y }$ containing the camouflaged object from $\hat { \mathbf { I } } _ { j }$ . We use the standard GAN loss as our objective. To train the discriminator, $D ,$ we minimize:

![](images/db2b6db6205836ba3768dc055c0b60ffdd19a8bd8d3f3e8951fec14f1d0de91a.jpg)  
Figure 3. Multi-view results. Multiple object views for selected scenes, camouflaged using our proposed model with four input views. The views shown here were held out and not provided to the network as input during training.

$$
\mathcal { L } _ { D } = - \mathbb { E } _ { y } [ \log D ( y ) ] - \mathbb { E } _ { \hat { y } } [ \log ( 1 - D ( \hat { y } ) ) ]\tag{6}
$$

where the expectation is taken over patches randomly sampled from a training batch. We implement our discriminator using the fully convolutional architecture of Isola et al. [24]. Our texturing function, meanwhile, minimizes:

$$
\mathcal { L } _ { a d v } = - \mathbb { E } _ { \hat { y } } [ \log D ( \hat { y } ) ]\tag{7}
$$

Self-supervised multi-view camouflage. We train our texturing function G (Eq. 2), which is fully defined by the image encoder E and the MLP T, by minimizing the combined losses:

$$
\begin{array} { r } { \mathscr { L } _ { G } = \mathscr { L } _ { p h o t o } + \lambda _ { a d v } \mathscr { L } _ { a d v } } \end{array}\tag{8}
$$

where $\lambda _ { a d v }$ controls the importance of the two losses.

If we were to train the model with only the input object, the discriminator would easily overfit, and our model would fail to obtain a learning signal. Moreover, the resulting texturing model would only be specialized to a single input shape, and may not generalize to others. To address both of these issues, we provide additional supervision to the model by training it to camouflage randomly augmented shapes at random positions, and from random subsets of views.

We sample object positions on the ground plane g, within a small radius proportional to the size of input object . We uniformly sample a position within the disk to determine the position for the object. In addition to randomly sampled locations, we also randomly scale the object within a range to add more diversity to training data. During training, we randomly select $N _ { i }$ input views and $N _ { r }$ rendering views without replacement from a pool of training images sampled from $\nu .$ We calculate $\mathcal { L } _ { p h o t o }$ on both $N _ { i }$ input views and $N _ { r }$ views while $\mathcal { L } _ { a d v }$ is calculated on $N _ { r }$ views.

## 4. Results

We compare our model to previous multi-view camouflage methods using cube shapes, as well as on complex animal and furniture shapes.

## 4.1. Dataset

We base our evaluation on the scene dataset of [33], placing objects at their predefined locations. Each scene contains 10-25 photos from different locations. During capturing, only background images are captured, with no actual object is placed in the scene. Camera parameters are estimated using structure from motion [47]. To support learning-based methods that take 4 input views, while still having a diverse evaluation set, we use 36 of the 37 scenes (removing one very small 6-view scene). In [33], their methods are only evaluated on cuboid shape, while our method can be adapted to arbitrary shape without any change to the model. To evaluate our method on complex shapes, we generate camouflage textures for a dataset of 49 animal meshes from [60]. We also provide a qualitative furniture shape from [11] (Fig. 1).

## 4.2. Implementation Details

For each scene, we reserve 1-3 images for testing (based on the total number of views in the scene). Following other work in neural textures [23], we train one network per scene. We train our models using the Adam optimizer [26] with a learning rate of $2 \times 1 0 ^ { - 4 }$ for the texturing function G and $1 0 ^ { - 4 }$ for the discriminator D. We use $\lambda _ { a d v } = 0 . 5$ in Eq. 8. We resize all images to be $3 8 4 \times 5 7 6$ and use square crops of $1 2 8 \times 1 2 8$ to calculate losses.

To ensure that our randomly chosen object locations are likely to be clearly visible from the cameras, we randomly sample object positions on the ground plane (the base of the cube in [33]). We allow these objects to be shifted at most 3 the cube’s length. During training, for each sample, we randomly select $N _ { i } = 4$ views as input views and render the object on another $N _ { r } = 2$ novel views. The model is trained with batch size of 8 for approximately 12k iterations. For evaluation, we place the object at the predefined position from [33] and render it in the reserved test views.

![](images/78a1851f5ea86a0c03b163a66e9633611861d6e44c984d057a4386d923995d93.jpg)  
(b) Qualitative results on animal shapes  
Figure 4. Comparison between methods for cuboids and complex shapes. We compare our method with previous approaches for the task of concealing (a) cuboids and (b) animal shapes. Our method produces objects with more coherent texture, with the 4-view mode filling in textures that tend to be occluded.

## 4.3. Experimental Settings

## 4.3.1 Cuboid shapes

We first evaluate our method using only cuboid shapes to compare with the state-of-the-art methods proposed in Owens et al. [33]. We compare our proposed 2-view and 4-view models with the following approaches:

Mean. The color for each 3D point is obtained by projecting it into all the views that observe it and taking the mean color at each pixel.

Iterative projection. These methods exploit the fact that an object can (trivially) be completely hidden from a single given viewpoint by back-projecting the image onto the object. When this is done, the object is also generally difficult to see from nearby viewpoints as well. In the Random method, the input images are selected in a random order, and each one is projected onto the object, coloring any surface point that has not yet been filled. In Greedy, the model samples the photos according to a heuristic that prioritizes viewpoints that observe the object head-on (instead of random sampling). Specifically, the photos are sorted based on the number of object faces that are observed from a direct angle (> 70<sup>◦</sup> with the viewing angle).

Example-based texture synthesis. These methods use Markov Random Fields (MRFs) [1, 16, 36] to perform example-based texture synthesis. These methods simultaneously minimize photoconsistency, as well as smoothness cost that penalizes unusual textures. The Boundary MRF model requires nodes within a face to have same labels, while Interior MRF does not.

## 4.3.2 Complex shapes

We also evaluated our model on a dataset containing 49 animal meshes [60]. Camouflaging these shapes presents unique challenges. In cuboids, the set of object points that each camera observes is often precisely the same, since each viewpoint sees at most 3 adjacent cube faces (out of 6 total). Therefore, it often suffices for a model to camouflage the most commonly-viewed object points with a single, coherent texture taken from one of the images, putting any conspicuous seams elsewhere on the object. In contrast, when the meshes have more complex geometry, each viewpoint sees a very different set of object points.

Since our model operates on arbitrary shapes, using these shapes requires no changes to the model. We trained our method with the animal shapes and placed the animal object at the same position as in the cube experiments. We adapt the simpler baseline methods of [33] to these shapes, however we note that the MRF-based synthesis methods assume a grid graph structure on each cube face, and hence cannot be adapted to complex shapes without significant changes.

<table><tr><td>Method</td><td>Confusion rate</td><td> $\mathbf { A v g . \ t i m e ^ { \textit { ( s ) } } }$ </td><td>Med. time (s)</td><td>n</td></tr><tr><td>Mean</td><td> $1 6 . 0 9 \% \pm 2 . 2 9$ </td><td> $4 . 8 2 \pm 0 . 3 7$ </td><td> $2 . 9 5 \pm 0 . 1 4$ </td><td>988</td></tr><tr><td>Random</td><td> $3 9 . 6 6 \% \pm 3 . 0 2$ </td><td> $7 . 6 3 \pm 0 . 5 0$ </td><td> $4 . 6 8 \pm 0 . 3 5$ </td><td>1011</td></tr><tr><td>Greedy</td><td> $4 0 . 3 2 \% \pm 2 . 9 6$ </td><td> $7 . 9 4 \pm 0 . 5 2$ </td><td> $4 . 7 2 \pm 0 . 3 6$ </td><td>1054</td></tr><tr><td>Boundary MRF [33]</td><td> $4 1 . 2 9 \% \pm 2 . 9 5$ </td><td> $8 . 5 0 \pm 0 . 5 1$ </td><td> $5 . 3 9 \pm 0 . 4 0$ </td><td>1068</td></tr><tr><td>Interior MRF [33]</td><td> $4 4 . 6 6 \% \pm 3 . 0 1$ </td><td> $8 . 1 9 \pm 0 . 5 1$ </td><td> $5 . 1 9 \pm 0 . 4 2$ </td><td>1048</td></tr><tr><td>Ours (2 views)</td><td> $\bar { \bf s } { \bf 1 . \bar { s } 8 \% } \bar { \pm } \bar { \bf 2 . 9 9 }$ </td><td> $\bar { \bf 9 . 1 9 \pm 0 . 5 1 }$ </td><td> $\mathbf { \bar { 6 . 4 6 } \bar { \pm } \bar { 0 . 4 2 } }$ </td><td>1074</td></tr><tr><td>Ours (4 views)</td><td> ${ \pm 3 . 9 5 \% } \pm 3 . 0 5$ </td><td> ${ \bf 9 . 2 9 \pm 0 . 5 7 }$ </td><td> ${ \bf 6 . 1 1 \pm 0 . 5 0 }$ </td><td>1025</td></tr></table>

Table 1. Perceptual study results with cubes. Higher numbers represent a better performance. We report the 95% confidence interval of these metrics.
<table><tr><td>Method</td><td>Confusion rate</td><td> $\mathbf { A v g . \ t i m e ^ { \textit { ( s ) } } }$ </td><td>Med. time (s)</td><td>n</td></tr><tr><td>Mean</td><td> $3 6 . 4 6 \% \pm 2 . 1 7$ </td><td> $6 . 3 9 \pm 0 . 3 0$ </td><td> $4 . 0 4 \pm 0 . 1 7$ </td><td>1898</td></tr><tr><td>Pixel-wise greedy</td><td> $5 0 . 4 3 \% \pm 2 . 2 0$ </td><td> $7 . 2 5 \pm 0 . 3 2$ </td><td> $4 . 7 3 \pm 0 . 2 0$ </td><td>1987</td></tr><tr><td>Random</td><td> $5 1 . 6 1 \% \pm 2 . 2 9$ </td><td> $7 . 8 1 \pm 0 . 3 6$ </td><td> $5 . 2 5 \pm 0 . 3 6$ </td><td>1831</td></tr><tr><td>Greedy</td><td> $5 2 . 5 0 \% \pm 2 . 1 8$ </td><td> $7 . 6 9 \pm 0 . 3 4$ </td><td> $5 . 1 3 \pm 0 . 2 5$ </td><td>2017</td></tr><tr><td>Ours (4 views)</td><td> $\mathbf { \bar { 6 1 . 9 3 \% } } \pm \bar { 2 } . \bar { 1 } 4$ </td><td> $\bar { \mathbf { 8 . 0 6 \pm 0 . 3 3 } }$ </td><td> $\bar { { \bf 5 . 6 6 } } \bar { { \bf \pm 0 . 2 7 } }$ </td><td>1970</td></tr></table>

Table 2. Perceptual study results on animal shapes. Higher numbers represent a better performance. We report the 95% confidence interval of these metrics.

Mean. As with cube experiment, we take the mean color from multiple input views as the simplest baseline.

Iterative projection. We use the same projection order selection strategy as in cube experiment. We determine whether a pixel is visible in the input views by using a raytriangle intersection test.

Pixel-wise greedy. Instead of projecting each input in sequential order, we choose the color for each pixel by selecting color from the input views that has largest view angle.

## 4.4. Perceptual Study

To evaluate the effectiveness of our method, we conduct a perceptual study. We generally follow the setup of [33], however we ask users to directly click on the camouflaged object [53], without presenting them with a second step to confirm that the object (or isn’t) present. This simplified the number of camouflaged objects that subjects see by a factor of two. We recruited 267 and 375 participants from Amazon Mechanical Turk for the perceptual study on cuboid and complex shapes, respectively, and ensured no participant attended both of the perceptual studies.

Each participant was shown one random image from the reserved images of each scene in a random order. The first 5 images that they were shown were part of a training exercise, and are not included in the final evaluation. We asked participants to search for the camouflaged object in the scene, and to click on it as soon as they found it. The object in the scene was camouflaged by a randomly chosen algorithm, and placed at the predefined position. After clicking on the image, the object outline was shown to the participant. We recorded whether the participant correctly clicked on the camouflaged object, and how long it took them to click. Each participant had one trial for each image and a maximum of 60s to find the camouflaged object.

Results on cuboid shapes. The perceptual study results on cuboid shapes are shown in Table 1. We report the confusion rate, average time, and median time measured over different methods. We found that our models significantly outperform the previous approaches on all metrics. To test for significance, we followed [33] and used a two-sided ttest for the confusion rate and a two-sided Mann-Whitney U test (with a 0.05 threshold for significance testing). We found that our method outperforms all the baseline methods significantly in the confusion rate metric. Both of our model variations outperform Interior MRF $( p < 2 \times 1 0 ^ { - 3 }$ and $p < 3 \times 1 0 ^ { - 5 } )$ . There was no significant difference between 2 and 4 views $( p = 0 . 2 8 )$ . In terms of time-to-click, our method also beats the two MRF-based methods. Compared with Boundary MRF, our method requires more time for participants to click the camouflaged object $\left( p = 0 . 0 0 2 4 \right.$ for 2 views and $p = 0 . 0 3 9$ for 4 views).

![](images/a82cd21680ca9750e61b5a71c116141fd5afbb9e6c2605a2183d215c02e7c5f7.jpg)

![](images/59475d3dc6be415a5a53029058c4412658e72c58cc4afeb01c4d3315f15b6365.jpg)

![](images/f7cdec3288f76ab924205b46e7fbd03b93fdb7918a014d643c37333749d696ff.jpg)  
(a) Real cube  
(b) With shadow  
(c) No shadow  
Figure 5. Effect of shadow on generated textures. We simulate the effect of shadows of the object in an indoor scene, using the reference object (a). Our model generates a texture with a shadow (b) by conditioning on composite images that contain the real shadow (but no real cube). (c) Result without shadow modeling.

Results on complex shapes. The perceptual study results on complex shapes are shown in Table 2. We found that our model obtained significantly better results than previous work on confusion rate. Our model also obtained significantly better results on the time-to-find metric. We found that in terms of confusion rate, our method with 4 input views is significantly better than the baseline methods, 9.42% better than Greedy method and 10.32% better than Random method. For time-to-click, our method also performs better than baseline methods compared with Greedy and Random.

## 4.5. Qualitative Results

We visualize our generated textures in several selected scenes for both cube shapes and animal shapes in Figure 3. We compare our method qualitatively with baseline methods from [33] in Figure 4. We found that our model obtained significantly more coherent textures than other approaches. The 2-view model has a failure case when none of the input views cover an occluded face, while the 4-view model is able to generally avoid this situation. We provide additional results in the supplement.

Effects of shadows. Placing an object in a real scene may create shadows. We ask how these shadows effect our model’s solution (Figure 5), exploiting the fact that these shadows are independent of the object’s texture and hence function similarly to other image content. In [33], photos with (and without) a real cube are taken from the same pose. We manually composite these paired images to produce an image without the real cube but with its real shadow. We then provide these images as conditioning input to our model, such that it incorporates the presence of the shadow into its camouflage solution. While our solution incorporates some of the shadowed region, the result is similar. Note that other lighting effects can be modeled as well (e.g., by compensating for known shading on the surface).

![](images/b0d7df42df395d85e7fb857767d0039f9afb5b29791cbf5166177b3de476d746.jpg)  
Figure 6. Ablations. We show how the choice of different components changes the quality of the camouflage texture.

## 4.6. Automated evaluation metrics

To help understand our proposed model, we perform an automated evaluation and compare with ablations:

• Adversarial loss: To evaluate the importance of $\mathcal { L } _ { a d v }$ , we set $\lambda _ { a d v }$ to 0 in Eq. 8. We evaluate the model performance with only $\mathcal { L } _ { p h o t o }$ used during training.

• Photoconsistency: We evaluate the importance of using all $N _ { i }$ input views in Eq. 5. The ablated model has $\mathcal { L } _ { p h o t o }$ only calculated on $N _ { r }$ rendering views during training.

• Architecture: We evaluate the importance of our pixelaligned feature representation. In lieu of this network, we use the feature encoder from pixelNeRF [55].

• Inpainting: Since inpainting methods cannot be directly applied to our task without substantial modifications, we combind several inpainting methods with the Greedy model. We selected several recent inpainting methods DeepFillv2 [56], LaMa [50], LDM [39] to inpaint the object shape in each view, then backproject this texture onto the 3D surface, using the geometry-based ordering from [33].

Evaluation metrics. To evaluate the ablated models, we use LPIPS [59] and SIFID metrics [43]. Since the background portion of the image remains unmodified, we use crops centered at the rendered camouflaged objects.

Results. Quantitative results are shown in Table 3 and qualitative results are in Figure 6. We found that our full 4-view model is the overall best-performing method. In particular, it significantly outperforms the 2-view model, which struggles when the viewpoints do not provide strong coverage from all angles (Fig. 6). We also found that the adversarial loss significantly improves performance. As can be seen in Fig. 6, the model without an adversarial loss fails to choose a coherent solution and instead appears to average all of the input views. The model that uses all views to compute photoconsistency tends to generate more realistic textures, perhaps due to the larger availability of samples. Compared with the pixelNeRF encoder, our model generates textures with higher fidelity, since it receives more detailed feature maps from encoder. We obtain better performance on LPIPS but find that this variation of the model achieves slightly better SIFID. This suggests that the architecture of our pixel-aligned features provides a modest improvement. Finally, we found that we significantly outperformed the inpainting and MRF-based methods.

<table><tr><td>Model</td><td>LPIPS↓</td><td>SIFID↓</td></tr><tr><td>Boundary MRF [33]</td><td>0.1228</td><td>0.0867</td></tr><tr><td>Interior MRF [33]</td><td>0.1185</td><td>0.0782</td></tr><tr><td>DeepFill v2 [56] + Projection [33]</td><td>0.1469</td><td>0.1245</td></tr><tr><td>LaMa [50] + Projection [33]</td><td>0.1263</td><td>0.1006</td></tr><tr><td>LDM [39] + Projection [33]</td><td>0.1305</td><td>0.0976</td></tr><tr><td>No  $\bar { \mathcal { L } } _ { a d v } ^ { \mathrm { ~ ~ - ~ } }$ </td><td>0.1064</td><td>0.0720</td></tr><tr><td>No  $\mathcal { L } _ { p h o t o }$  on input views</td><td>0.1131</td><td>0.0856</td></tr><tr><td>With pixelNeRF encoder [55]</td><td>0.1047</td><td>0.0712</td></tr><tr><td>Ours (2 views)</td><td>0.1079</td><td>0.0754</td></tr><tr><td>Ours (4 views)</td><td>0.1034</td><td>0.0714</td></tr></table>

Table 3. Evaluation with automated metrics. We compare our method to other approaches, and perform ablations.

## 5. Discussion

We proposed a method for generating textures to conceal a 3D object within a scene. Our method can handle diverse and complex 3D shapes and significantly outperforms previous work in a perceptual study. We see our work as a step toward developing learning-based camouflage models. Additionally, the animal kingdom has a range of powerful camouflage strategies, such as disruptive coloration and mimicry, that cleverly fool the visual system and may require new learning methods to capture.

Limitations. As in other camouflage work [33], we do not address the problem of physically creating the camouflaged object, and therefore do not systematically address practicalities like lighting and occlusion.

Ethics. The research presented in this paper has the potential to contribute to useful applications, particularly to hiding unsightly objects, such as solar panels and utility boxes. However, it also has the potential to be used for negative applications, such as hiding nefarious military equipment and intrusive surveillance cameras.

Acknowledgements. We thank Justin Johnson, Richard Higgins, Karan Desai, Gaurav Kaul, Jitendra Malik, and Derya Akkaynak for the helpful discussions and feedback. This work was supported in part by an NSF GRFP for JC.

## References

[1] Aseem Agarwala, Mira Dontcheva, Maneesh Agrawala, Steven Drucker, Alex Colburn, Brian Curless, David Salesin, and Michael Cohen. Interactive digital photomontage. In ACM SIGGRAPH 2004 Papers, pages 294–302. 2004. 6

[2] Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples. In International conference on machine learning, pages 284–293. PMLR, 2018. 2

[3] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Trans. Graph., 28(3):24, 2009. 3

[4] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 417–424, 2000. 3

[5] Sai Bi, Nima Khademi Kalantari, and Ravi Ramamoorthi. Patch-based optimization for image-based texture mapping. ACM Trans. Graph., 36(4):106–1, 2017. 2

[6] Tom B Brown, Dandelion Mane, Aurko Roy, Mart´ ´ın Abadi, and Justin Gilmer. Adversarial patch. arXiv preprint arXiv:1712.09665, 2017. 2

[7] Joshua Callaghan. Public art projects, 2016. 1

[8] Hala Lamdouar Charig Yang, Erika Lu, Andrew Zisserman, and Weidi Xie. Self-supervised video object segmentation by motion grouping. 2021. 2

[9] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. arXiv preprint arXiv:2103.15595, 2021. 3, 4

[10] Hung-Kuo Chu, Wei-Hsin Hsu, Niloy J Mitra, Daniel Cohen-Or, Tien-Tsin Wong, and Tong-Yee Lee. Camouflage images. ACM Trans. Graph., 29(4):51–1, 2010. 2

[11] Jasmine Collins, Shubham Goel, Achleshwar Luthra, Leon Xu, Kenan Deng, Xi Zhang, Tomas F Yago Vicente, Himanshu Arora, Thomas Dideriksen, Matthieu Guillaumin, and Jitendra Malik. Abo: Dataset and benchmarks for real-world 3d object understanding. arXiv preprint arXiv:2110.06199, 2021. 5

[12] Hugh Bamford Cott. Adaptive coloration in animals. 1940. 2

[13] Richard Dawkins et al. The blind watchmaker: Why the evidence of evolution reveals a universe without design. WW Norton & Company, 1996. 2

[14] Alexei A Efros and Thomas K Leung. Texture synthesis by non-parametric sampling. In Proceedings of the seventh IEEE international conference on computer vision, volume 2, pages 1033–1038. IEEE, 1999. 3

[15] Deng-Ping Fan, Ge-Peng Ji, Guolei Sun, Ming-Ming Cheng, Jianbing Shen, and Ling Shao. Camouflaged object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2777–2787, 2020. 2

[16] William T Freeman, Thouis R Jones, and Egon C Pasztor. Example-based super-resolution. IEEE Computer graphics and Applications, 22(2):56–65, 2002. 6

[17] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016. 3, 4

[18] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014. 2

[19] Roger Hanlon. Cephalopod dynamic camouflage. Current Biology, 17(11):R400–R404, 2007. 2

[20] Bharath Hariharan, Pablo Arbelaez, Ross Girshick, and Ji-´ tendra Malik. Hypercolumns for object segmentation and fine-grained localization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 447–456, 2015. 2, 4

[21] James Hays and Alexei A Efros. Scene completion using millions of photographs. ACM Transactions on Graphics (TOG), 26(3):4–es, 2007. 3

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 4

[23] Philipp Henzler, Niloy J Mitra, , and Tobias Ritschel. Learning a neural 3d texture space from 2d exemplars. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 2, 3, 4, 5

[24] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017. 5

[25] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision, 2016. 4

[26] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 5

[27] Hala Lamdouar, Charig Yang, Weidi Xie, and Andrew Zisserman. Betrayed by motion: Camouflaged object discovery via motion segmentation. In Proceedings of the Asian Conference on Computer Vision, 2020. 2

[28] Trung-Nghia Le, Yubo Cao, Tan-Cong Nguyen, Minh-Quan Le, Khanh-Duy Nguyen, Thanh-Toan Do, Minh-Triet Tran, and Tam V Nguyen. Camouflaged instance segmentation in-the-wild: Dataset and benchmark suite. arXiv preprint arXiv:2103.17123, 2, 2021. 2

[29] Rob Matheson. Solar panels get a face-lift with custom designs, 2017. 1

[30] Sami Merilaita, Nicholas E Scott-Samuel, and Innes C Cuthill. How camouflage works. Philosophical Transactions of the Royal Society B: Biological Sciences, 372(1724):20160341, 2017. 1

[31] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European conference on computer vision, pages 405–421. Springer, 2020. 3, 4

[32] Michael Oechsle, Lars Mescheder, Michael Niemeyer, Thilo Strauss, and Andreas Geiger. Texture fields: Learning texture representations in function space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4531–4540, 2019. 2, 3, 4

[33] Andrew Owens, Connelly Barnes, Alex Flint, Hanumant Singh, and William Freeman. Camouflaging an object from many viewpoints. In CVPR, 2014. 1, 2, 3, 5, 6, 7, 8

[34] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2536–2544, 2016. 2, 3

[35] Tiziano Portenier, Siavash Arjomand Bigdeli, and Orcun Goksel. Gramgan: Deep 3d texture synthesis from 2d exemplars. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6994–7004. Curran Associates, Inc., 2020. 2, 3

[36] Yael Pritch, Eitam Kav-Venaki, and Shmuel Peleg. Shiftmap image editing. In 2009 IEEE 12th international conference on computer vision, pages 151–158. IEEE, 2009. 6

[37] Nikhila Ravi, Jeremy Reizenstein, David Novotny, Taylor Gordon, Wan-Yen Lo, Justin Johnson, and Georgia Gkioxari. Accelerating 3d deep learning with pytorch3d. arXiv:2007.08501, 2020. 4

[38] Craig Reynolds. Interactive evolution of camouflage. Artificial life, 17(2):123–136, 2011. 2

[39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models, 2021. 8

[40] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi, editors, Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pages 234– 241, Cham, 2015. Springer International Publishing. 4

[41] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015. 4

[42] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2304–2314, 2019. 2

[43] Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli. Singan: Learning a generative model from a single natural image. In Proceedings of the IEEE International Conference on Computer Vision, pages 4570–4580, 2019. 8

[44] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, 2015. 4

[45] Karl Sims. Evolving 3d morphology and behavior by competition. Artificial life, 1(4):353–372, 1994. 2

[46] Vincent Sitzmann, Michael Zollhofer, and Gordon Wet-¨ zstein. Scene representation networks: Continuous 3dstructure-aware neural scene representations. arXiv preprint arXiv:1906.01618, 2019. 3

[47] Noah Snavely, Steven Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. acm trans graph 25(3):835-846. ACM Trans. Graph., 25:835–846, 07 2006. 5

[48] Martin Stevens and Sami Merilaita. Animal camouflage: mechanisms and function. Cambridge University Press, 2011. 2

[49] Jack Stewart. Tesla unveils its new line of camouflaged solar panels, 2016. 1

[50] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. arXiv preprint arXiv:2109.07161, 2021. 8

[51] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013. 2

[52] Laszlo Talas, John G Fennell, Karin Kjernsmo, Innes C Cuthill, Nicholas E Scott-Samuel, and Roland J Baddeley. Camogan: Evolving optimum camouflage with generative adversarial networks. Methods in Ecology and Evolution, 11(2):240–247, 2019. 1, 2

[53] Jolyon Troscianko, Jared Wilson-Aggarwal, David Griffiths, Claire N Spottiswoode, and Martin Stevens. Relative advantages of dichromatic and trichromatic color vision in camouflage breaking. Behavioral Ecology, 28(2):556–564, 2017. 2, 7

[54] Jinnan Yan, Trung-Nghia Le, Khanh-Duy Nguyen, Minh-Triet Tran, Thanh-Toan Do, and Tam V Nguyen. Mirrornet: Bio-inspired camouflaged object segmentation. IEEE Access, 9:43290–43300, 2021. 2

[55] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4578–4587, 2021. 2, 3, 4, 8

[56] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Free-form image inpainting with gated convolution. arXiv preprint arXiv:1806.03589, 2018. 8

[57] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5505–5514, 2018. 3

[58] Qing Zhang, Gelin Yin, Yongwei Nie, and Wei-Shi Zheng. Deep camouflage images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 12845– 12852, 2020. 2

[59] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 8

[60] Silvia Zuffi, Angjoo Kanazawa, David Jacobs, and Michael J. Black. 3D menagerie: Modeling the 3D shape and pose of animals. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), July 2017. 5, 6