pub mod layers;

pub mod learning {
    use crate::network::layers::layers_structure::{Layer, LayerType};
    use libm::exp;
    use ndarray::prelude::*;
    use rand::prelude::*;

    #[derive(Debug)]
    pub struct Learning {
        pub input: Vec<Vec<(Array1<f32>, Array1<f32>)>>,
        pub sizes: Vec<usize>,
        pub mini_batch_size: usize,
        pub epochs: usize,
        pub learning_rate: f32,
        pub lambda: f32,
        pub layers: Vec<Layer>,
    }

    impl Learning {
        /*fn get_image_scale(
                    image: &mut GrayImage,
                    mut scale: &mut Vec<f64>,
                    dimensions: (u32, u32),
                ) {
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

                // fn next_image(counter: &mut usize) -> GrayImage {
                //     let path = from_fn(|| Some(String::new()))
                //         .enumerate()
                //         .map(|mut x| {
                //             x.1 = counter.to_string() + ".jpg";
                //             x.1
                //         })
                //         .take(1)
                //         .collect::<String>();
                //     *counter += 1;
                //     open(path).unwrap().to_luma8()
                // }
        */
        pub fn new(
            slice: Vec<f32>,
            answer: Vec<f32>,
            layers: Vec<usize>,
            mini_batch_size: usize,
            learning_rate: f32,
            lambda: f32,
            epochs: usize,
        ) -> Self {
            Learning {
                input: {
                    let mut input_data: Vec<Vec<(Array1<f32>, Array1<f32>)>> = vec![];
                    for a in slice.chunks(784).zip(answer.chunks(10)) {
                        input_data.push({
                            let mut b: Vec<(Array1<f32>, Array1<f32>)> = vec![];
                            for _ in 0..mini_batch_size {
                                b.push((
                                    Array1::from_vec(a.0.to_vec()),
                                    Array1::from_vec(a.1.to_vec()),
                                ));
                            }
                            b
                        })
                    }
                    input_data
                },
                layers: {
                    let mut cur_size = layers.iter();
                    cur_size.next().unwrap();
                    let mut prev_size = layers.iter();
                    let mut a: Vec<Layer> = Vec::with_capacity(layers.len() - 1);
                    for _ in 0..layers.len() - 1 {
                        a.push({
                            Layer::new(
                                *prev_size.next().unwrap(),
                                *cur_size.next().unwrap(),
                                LayerType::Sigmoid,
                            )
                        });
                    }
                    a
                },
                sizes: layers,
                mini_batch_size,
                epochs,
                learning_rate,
                lambda,
            }
        }

        pub fn stochastic_gradient_descent(&mut self) {
            for _ in 0..self.epochs {
                self.input.shuffle(&mut thread_rng());
                for mini_batch in self.input.clone().iter() {
                    self.update_mini_batch(mini_batch, self.learning_rate, self.lambda, 50_000);
                    println!("Epoch complete, proceed");
                    println!("Accuracy: {}", self.check_accuracy());
                }
            }
        }

        pub fn update_mini_batch(
            &mut self,
            mini_batch: &Vec<(Array1<f32>, Array1<f32>)>,
            learning_rate: f32,
            lambda: f32,
            train_size: usize,
        ) {
            let mut nabla_w = Array1::<Array2<f32>>::default(self.layers.len());
            let mut nabla_b = Array1::<Array2<f32>>::default(self.layers.len());
            for (x, y) in mini_batch {
                let (delta_w, delta_b) = self.backprop(x, y);
                nabla_b = nabla_b
                    .into_iter()
                    .zip(delta_b)
                    .map(|x| x.0 + x.1)
                    .collect();
                nabla_w = nabla_w
                    .into_iter()
                    .zip(delta_w)
                    .map(|x| x.0 + x.1)
                    .collect()
            }
            self.layers.iter_mut().zip(nabla_w).for_each(|x| {
                x.0.weights = (1.0 - learning_rate * (lambda / train_size as f32)) * &x.0.weights
                    - (learning_rate / mini_batch.len() as f32) * &x.1;
            });
            self.layers.iter_mut().zip(nabla_b).for_each(|x| {
                x.0.biases = &x.0.biases
                    - (learning_rate / mini_batch.len() as f32)
                        * &Array1::from_vec(x.1.into_raw_vec());
            });
        }

        fn feedforward(&self, input: &Array1<f32>) -> Array1<f32> {
            let mut a: Array1<f32> = input.clone();
            self.layers.iter().for_each(|x| {
                a = x.weights.dot(&a) + &x.biases;
            });
            a
        }

        /// The output is nabla_w and nabla_b
        pub fn backprop(
            &mut self,
            x: &Array1<f32>,
            y: &Array1<f32>,
        ) -> (Array1<Array2<f32>>, Array1<Array2<f32>>) {
            let mut nabla_b: Array1<Array2<f32>> = Array1::default(self.layers.len());
            let mut nabla_w: Array1<Array2<f32>> = Array1::default(self.layers.len());
            let mut activations: Vec<Array1<f32>> = Vec::with_capacity(self.layers.len());
            let mut activation = x.clone();
            activations.push(activation.to_owned());
            for it in self.layers.iter_mut() {
                it.z_vec = it.weights.dot(&activation) + &it.biases;
                activation = Self::sigmoid(&it.z_vec.view());
                activations.push(activation.to_owned());
            }
            let mut delta: Array2<f32> = self
                .layers
                .last()
                .unwrap()
                .delta(activations.last().unwrap(), y);
            println!("{:?}", delta);
            nabla_b[self.layers.len() - 1] = delta.clone();
            let a = &Array2::from_shape_vec(
                (1, activations.last().unwrap().len()),
                activations.last().unwrap().to_vec(),
            )
            .unwrap();
            nabla_w[self.layers.len() - 1] = delta.dot(&a.t());
            println!("{:?}", nabla_w[self.layers.len() - 1]);
            for l in (1..self.layers.len()).rev() {
                delta = self.layers[l].weights.t().dot(&delta);
                delta = delta * Self::sigmoid_prime(&self.layers[l - 1].z_vec.view()).t();
                nabla_b[l] = delta.clone();
                let a = &Array2::from_shape_vec((1, activations[l].len()), activations[l].to_vec())
                    .unwrap();
                nabla_w[l] = delta.dot(&a.t());
                println!("{:?}", nabla_w[l]);
            }
            (nabla_w, nabla_b)
        }

        pub(crate) fn arg_max(array: &Array1<f32>) -> usize {
            array
                .iter()
                .enumerate()
                .fold((0, array[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                })
                .0
        }

        fn check_accuracy(&self) -> usize {
            self.input
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|y| {
                            let row = self.feedforward(&y.0);
                            let max_idx = Learning::arg_max(&row);
                            (Learning::arg_max(&y.1) == max_idx) as usize
                        })
                        .sum::<usize>()
                })
                .sum()
        }
        /* pub fn backpropagation_calculus(&mut self) {
                    let mut input_layer = vec![self.layer_input.input.clone()];
                    let mut layer_ = self.layer_bias.clone();
                    layer_.iter_mut().enumerate().for_each(|x| {
                        x.1.fill_input(input_layer.get(x.0).unwrap().clone());
                        x.1.sigmoid();
                        input_layer.push(Arc::new(
                            x.1.nodes.iter().map(|x| x.output).collect::<Vec<f64>>(),
                        ));
                    });
                    self.layer_digit.fill_input(Arc::new(
                        self.layer_bias
                            .last()
                            .unwrap()
                            .nodes
                            .iter()
                            .map(|x| x.output)
                            .collect::<Vec<f64>>(),
                    ));
                    self.layer_digit.sigmoid();
                    self.layer_digit.change_weight();
                    let temp = (*self.layer_bias.last().unwrap()).clone();
                    self.layer_bias
                        .last_mut()
                        .unwrap()
                        .error_generate(&temp, &self.layer_digit);
                    let mut input_ = input_layer.last_mut().unwrap().as_ref().clone();
                    Learning::get_image_scale(
                        &mut Learning::next_image(&mut self.counter),
                        &mut input_,
                        (0, 0),
                    );
                    //self.layer_input = LayerInput::new(&mut input_layer.last().unwrap());
                }
        */
        fn sigmoid(z: &ArrayView1<f32>) -> Array1<f32> {
            let mut a = Array1::zeros(z.raw_dim());
            for i in a.iter_mut().enumerate() {
                *i.1 = 1.0 / (1.0 + exp(-z[i.0] as f64) as f32);
            }
            a.t().to_owned()
        }
        fn sigmoid_prime(z: &ArrayView1<f32>) -> Array1<f32> {
            let a = Self::sigmoid(z);
            &a * (1.0 - &a)
        }
    }
    #[test]
    fn check_max() {
        assert_eq!(
            4,
            Learning::arg_max(&Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 4.0]))
        );
    }
}
