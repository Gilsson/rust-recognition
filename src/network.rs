pub mod layers;

pub mod learning {
    use crate::network::layers::layers_structure::{
        BackPropagation, Layer, LayerBias, LayerDigit, LayerInput,
    };
    use crate::network::layers::node::nodes::NodeIO;
    use image::{open, DynamicImage, GrayImage, Luma};
    use libm::fabsf;
    use std::borrow::Borrow;
    use std::iter::from_fn;
    use std::ops::Deref;
    use std::str::FromStr;

    #[derive(Debug)]
    pub struct Learning {
        image: GrayImage,
        pub layer_input: LayerInput,
        pub layer_bias: Vec<LayerBias>,
        pub layer_digit: LayerDigit,
    }

    impl Learning {
        fn get_image_scale(image: &mut GrayImage, scale: &mut Vec<f64>, dimensions: (u32, u32)) {
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
                Learning::get_image_scale(
                    image,
                    scale,
                    (dimensions.0 + image.width() / 7, dimensions.1),
                );
            } else {
                Learning::get_image_scale(image, scale, (0, dimensions.1 + image.width() / 7));
            }
        }

        fn next_image() -> GrayImage {
            let path = from_fn(|| Some(String::new()))
                .enumerate()
                .map(|mut x| {
                    x.1 = x.0.to_string() + ".jpg";
                    x.1
                })
                .take(1)
                .collect::<String>();
            open(path).unwrap().to_luma8()
        }

        pub fn new(layers: Vec<usize>, path: String) -> Learning {
            let number = path.get(0..1).unwrap();
            let i = usize::from_str(number);
            let mut image: GrayImage = open(path).unwrap().to_luma8();
            let mut input: Vec<f64> = Vec::new();
            let mut iter1 = layers.iter();
            let mut iter2 = layers.iter();
            iter2.next();

            Learning::get_image_scale(&mut image, &mut input, (0, 0));
            Learning {
                image,
                layer_input: LayerInput::new(input),
                layer_bias: from_fn(|| {
                    Some(LayerBias::new(
                        *iter1.next().unwrap(),
                        *iter2.next().unwrap(),
                    ))
                })
                .take(2)
                .collect(),
                layer_digit: LayerDigit::new(*layers.last().unwrap(), 10),
            }
        }
        pub fn backpropagation_calculus(&mut self) {
            let mut input = self.layer_input.input.clone();
            self.layer_bias.iter_mut().enumerate().for_each(|x| {
                x.1.fill_input(&input);
                input = x.1.nodes.iter().map(|x| x.output).collect::<Vec<f64>>();
            });
            self.layer_bias.iter_mut().for_each(|x| x.sigmoid());
            self.layer_digit.fill_input(
                &self
                    .layer_bias
                    .last()
                    .unwrap()
                    .nodes
                    .iter()
                    .map(|x| x.output)
                    .collect::<Vec<f64>>(),
            );
            self.layer_digit.sigmoid();
            let mut input = &self.layer_digit.nodes;
            let mut errors = self.layer_digit.get_errors();
            self.layer_bias.reverse();
            self.layer_bias.iter_mut().for_each(|x| {
                x.error_generate(input, errors);
                input = x.get_nodes();
                errors = x.get_errors();
            });
        }
    }
}
