# YOLO-object-detection-in-java
A java based template for streaming video based object detection using only YOLO weights
  
 ## What this project is about :

While there is lot of focus in doing deep learning based image classification and object detection models in python and consequently there are numerous blogs showing how to do the same there is little information on how to use JAVA based packages like OpenCV in a standalone manner to do image classification and recognition tasks ,this project focuses on a step by step approach on doing a video stream analysis using nothing but JAVA , OpenCV package in java and YOLO . Keep in mind we do not use tensorflow/Darkflow/Darknet  in this project . In next sections  we go thru in detail on what is object detection , what is YOLO and how to implement YOLO using OpenCV and JAVA . We will follow it up with a sample JAVA code using YOLO models to detect objects in Video stream explained in Detail  



# Why we not use Python as our language for this project :

We don’t use python simply because
1.	As already mentioned there are multiple blogs , projects, repos showcasing how to use python with different deep learning frameworks like Torch ,Keras ,Caffee, Tensorflow for image recognition
2.	Image detection and object recognition is an upcoming field in area of digitalization , however most systems and industries that need to be digitalized run on java based platforms consequently it might be difficult for them use languages like python in their existing architecture.
3.	Though there are hybrid architectures that attempt to leverage both older platforms with newer frameworks like tensorflow along with python these types of architectures often lead to speed scalability issues thus making project deployments and maintainace difficult 
