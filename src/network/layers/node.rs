pub mod nodes {
    use rand::Rng;
    use std::iter::from_fn;

    trait Node {}

    pub trait RandomGeneration {
        fn generate_number(upper_case: f64) -> f64 {
            let mut rng_gen = rand::thread_rng();
            rng_gen.gen_range(0.0..upper_case)
        }
        fn generate_range(size: usize, upper_case: f64) -> Option<Vec<f64>> {
            let mut rng_gen = rand::thread_rng();
            let matching = if upper_case > 0.0 && upper_case < 0.1 {
                0
            } else {
                1
            };
            match size {
                0 => None,
                _ => match matching {
                    0 => Some(from_fn(|| Some(rng_gen.gen::<f64>())).take(size).collect()),
                    _ => Some(
                        from_fn(|| Some(rng_gen.gen_range(0.0..upper_case)))
                            .take(size)
                            .collect(),
                    ),
                },
            }
        }
    }
    #[derive(Debug, Default)]
    pub struct NodeIO {
        pub input: Vec<f64>,
        pub weight: Vec<f64>,
        pub bias: f64,
        pub output: f64,
    }

    impl RandomGeneration for NodeIO {}

    impl NodeIO {
        pub fn new(size: usize) -> NodeIO {
            NodeIO {
                input: {
                    let mut input = Vec::<f64>::new();
                    input.resize(size, 0.0);
                    input
                },
                weight: from_fn(|| Some(NodeIO::generate_number(2.0)))
                    .take(size)
                    .collect(),
                bias: NodeIO::generate_number(2.0),
                output: 0.0,
            }
        }
    }
}
