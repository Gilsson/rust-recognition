pub mod nodes {
    use libm::exp;
    use rand::prelude::*;
    use rand_distr::Normal;
    use std::cell::RefCell;
    use std::cmp::max;
    use std::iter::from_fn;
    use std::sync::Arc;

    #[derive(Debug, Clone)]
    enum OutType {
        Sigmoid,
        ReLU,
    }

    /*impl RandomGeneration for Node {}

    impl Node {
        pub fn new(prev_layer_size: usize, out_type: OutType) -> Node {
            Node {
                out_type,
                input: Arc::new({
                    let mut input = Vec::<f32>::new();
                    input.resize(prev_layer_size, 0.0);
                    input
                }),
                weight: Box::new(
                    Node::generate_range(
                        prev_layer_size,
                        0.0,
                        1.0 / f32::sqrt(prev_layer_size as f32),
                    )
                    .unwrap(),
                ),
                bias: Node::generate_number(0.0, 1.0),
                error: Box::new(from_fn(|| Some(0.0)).take(prev_layer_size).collect()),
                output: 0.0,
                z: 0.0,
            }
        }
        fn init(&mut self) {
            self.z = {
                let mut z = 0.0;
                for it in self.input.iter().zip(self.weight.iter()) {
                    let (input, weight) = it;
                    z += *input * *weight + self.bias;
                }
                z
            };
            match self.out_type {
                OutType::Sigmoid => self.output = 1.0 / (1.0 + exp(-self.z as f64)) as f32,
                OutType::ReLU => self.output = f32::max(0.0, self.z),
            }
        }
    }*/
}
