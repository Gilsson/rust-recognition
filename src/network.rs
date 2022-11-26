pub mod layers;

pub mod learning {
<<<<<<< HEAD
    use crate::network::layers::layers_structure::{Layer, LayerBias, LayerDigit, LayerInput};
    use image::{open, DynamicImage, GrayImage, Luma};
    use libm::fabsf;
    use std::iter::from_fn;
    use std::rc::Rc;

    #[derive(Debug)]
    pub struct Learning {
        image: GrayImage,
        layer_input: Rc<LayerInput>,
        layer_bias: Rc<Vec<LayerBias>>,
        layer_digit: Rc<LayerDigit>,
    }

    impl Learning {
        fn check_image_field(image: &mut GrayImage, scale: &mut Vec<f64>, dimensions: (u32, u32)) {
            if dimensions.1 >= image.width() {
                return;
            }
            let y: u32 = dimensions.1;
            for x in dimensions.0..image.width() {
                //let pixel = Luma([rng.gen::<u8>() as u8]);
                //image.put_pixel(x, y, pixel);
                scale.push(fabsf(
                    ((*image.get_pixel(x, y).0.first().unwrap()) as f32 / 255.0 - 1.0) as f32,
                ) as f64);
            }
            if dimensions.0 < image.height() - image.width() / 7 {
                Learning::check_image_field(
                    image,
                    scale,
                    (dimensions.0 + image.width() / 7, dimensions.1),
                );
            } else {
                Learning::check_image_field(image, scale, (0, dimensions.1 + image.width() / 7));
            }
        }
        
        pub fn new(layers: Vec<usize>) -> Vec<Learning> {
            let mut path = Vec::<String>::new();
            path = from_fn(|| Some(String::new()))
                .enumerate()
                .map(|mut x| {
                    x.1 = x.0.to_string() + ".jpg";
                    x.1
                })
                .take(5)
                .collect();
            //let path = "{i}.jpg";
            let mut image: Vec<GrayImage> = path
                .iter()
                .map(|x| open(x).unwrap().to_luma8())
                .collect::<Vec<GrayImage>>();
            let mut input: Vec<f64> = Vec::new();
            let mut iter = layers.iter();
            image
                .into_iter()
                .map(|mut x| {
                    Learning::check_image_field(&mut x, &mut input, (0, 0));
                    x
                })
                .take(5)
                .collect::<Vec<GrayImage>>();
            image
                .into_iter()
                .map(|x| Learning {
                    image: x,
                    layer_input: Rc::new(LayerInput::new(input)),
                    layer_bias: Rc::new({
                        from_fn(|| {
                            Some(LayerBias::new(
                                *layers.first().unwrap(),
                                *iter.next().unwrap(),
                            ))
                        })
                        .take(layers.len())
                        .collect()
                    }),
                    layer_digit: Rc::new(LayerDigit::new(*layers.last().unwrap(), 10)),
                })
                .collect::<Vec<Learning>>()
        }
        pub fn backpropagation_calculus(&mut self) {}
=======
    use rand::Rng;

    
    struct Learning {
        
>>>>>>> 2b34d59 (first commit)
    }
}
