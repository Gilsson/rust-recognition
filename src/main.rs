pub mod network;

use crate::network::learning::Learning;
use mnist::MnistBuilder;

fn main() {
    let mnist = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(70_000)
        .validation_set_length(0)
        .test_set_length(0)
        .finalize();
    let slice: Vec<f32> = mnist
        .trn_img
        .into_iter()
        .map(|x| x as f32 / 256.0)
        .collect();
    let answer: Vec<f32> = mnist.trn_lbl.into_iter().map(|x| x as f32).collect();
    //println!("{:?}", mnist.trn_lbl);
    let mut network = Learning::new(
        slice,
        answer,
        vec![784, 40, 10],
        2, /* usize */
        0.5,
        2.0, /* f32 */
        50,  /* usize */
    );
    unsafe {
        network.stochastic_gradient_descent();
    }
    //println!("{:?}", network);
    //let mut file = File::create("test.txt").expect("Some message");
    /* writeln!(file, "{:?}", network.layer_bias.get(1).unwrap()).expect("Some");
    for _ in 0..=5 {
        network.backpropagation_calculus();
        writeln!(file, "\n \n \n \n \n").expect("Some");
        writeln!(file, "{:?}", network.layer_bias.get(1).unwrap()).expect("Some");
    }

    writeln!(file, "\n \n \n \n \n").expect("Some");
    writeln!(file, "{:?}", network.layer_bias.get(1).unwrap()).expect("Some");
    //network.into_iter().map(|mut x| x.backpropagation_calculus()).take(1).collect();*/
}
/*let mut layer = LayerBias::new(16);
let x = 28usize;
let y = 28usize;
let path = "20.jpg";
let mut bwimage = open(path).unwrap().to_luma8();
let mut input: Vec<f64> = Vec::new();
check_image_field(&mut bwimage, &mut input, (0, 0));
layer.fill_input(input);
layer.sigmoid();
println!("{:?}", layer);
let mut layer = Sigmoid(Box::new());
let mut layer_sigmoid: Layers = Layers::new(Layers::LayerSigmoid { Def }, 15);
layer_sigmoid.fill_elements(&input);
let mut digit_layer = LayerSigmod::create_layer(&layer_sigmoid);
println!("{}", digit_layer.get_most_valuable_digit());
loop {
    layer_sigmoid.fill_elements(&input);
    layer_sigmoid.change_weights(&digit_layer);
    let mut digit_layer = LayerDigits::create_layer(&layer_sigmoid);
    println!("{:?}", digit_layer);
    println!("{}", digit_layer.get_most_valuable_digit());
}
bwimage.save("test.png").unwrap();
// Save the image as ???fractal.png???, the format is deduced from the path
 */

/*fn check_image_field(image: &mut GrayImage, scale: &mut Vec<f64>, dimensions: (u32, u32)) {
    if dimensions.1 >= image.width() {
        return;
    }
    let y: u32 = dimensions.1;
    for x in dimensions.0..image.width() {
        //let pixel = Luma([rng.gen::<u8>() as u8]);
        //image.put_pixel(x, y, pixel);
        scale.push(
            fabsf(((*image.get_pixel(x, y).0.first().unwrap()) as f32 / 255.0 - 1.0) as f32) as f64,
        );
    }
    if dimensions.0 < image.height() - image.width() / 7 {
        check_image_field(
            image,
            scale,
            (dimensions.0 + image.width() / 7, dimensions.1),
        );
    } else {
        check_image_field(image, scale, (0, dimensions.1 + image.width() / 7));
    }
}
*/
//#[test]
//fn test_digit() {
//    let layer = LayerDigits {
//        input: vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 0.0, 9.0, 0.0],
//    };
//    assert_eq!(layer.get_most_valuable_digit(), 8);
//}
