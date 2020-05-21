# Single Image Haze Removal
>This is a NON OFICIAL C++ implementation for He et al (2009), ["Single Image Haze Removal Using Dark Channel Prior"](http://kaiminghe.com/publications/cvpr09.pdf)

## Requirements

- [OpenCV](https://opencv.org/)

## Build and Usage

> Set the OpenCV_DIR at [CMakeLists.txt](CMakeLists.txt) to point to your OpenCV path.

Run the following commands to build:

```bash
cd build                           # Enter build folder
cmake ..                           # Configure environment
make -j4                           # Build executable
./bin/haze_removal <IMAGE_PATH>    # Run with an input image
```

## References

This implementation is based on [linzhi's](https://github.com/linzhi/dehazing) and [ rexledesma's](https://github.com/rexledesma/Haze-Removal).
