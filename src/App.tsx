import { useEffect, useState } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import {
  Box,
  Center,
  Flex,
  Heading,
  Image as ImageChakra,
} from "@chakra-ui/react";
import { Button } from "./components/ui/button";
import { HiUpload } from "react-icons/hi";
import { FileUploadRoot, FileUploadTrigger } from "./components/ui/file-upload";
import { Tag } from "./components/ui/tag";
import { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";

// Definisi custom layer dalam TypeScript
// class Normalization extends tf.layers.Layer {
//   mean: number;
//   variance: number;

//   constructor(config: LayerArgs & { mean?: number; variance?: number }) {
//     super(config);
//     this.mean = config.mean || 0; // Default mean
//     this.variance = config.variance || 1; // Default variance
//   }

//   call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
//     const x = Array.isArray(inputs) ? inputs[0] : inputs;
//     return tf.tidy(() => {
//       return x.sub(tf.scalar(this.mean)).div(tf.sqrt(tf.scalar(this.variance)));
//     });
//   }

//   static get className(): string {
//     return "Normalization";
//   }
// }

function App() {
  const [selectedImage, setSelectedImage] = useState<null | string>(null);
  const [model, setModel] = useState<null | tf.LayersModel>(null);
  const [predictedlabel, setPredictedLabel] = useState<null | string>();
  // (tf.serialization as any).registerClass(Normalization);

  const loadModel = async () => {
    try {
      const model = await tf.loadLayersModel("/tm-my-image-model/model.json");
      console.log("Model loaded successfully!");
      setModel(model);
    } catch (error) {
      console.error("Failed to load model:", error);
    }
  };

  useEffect(() => {
    loadModel();
  }, []);

  const handleImageUpload = (event: any) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        const imageDataUrl = reader.result as string;
        setSelectedImage(imageDataUrl);
      };

      // Membaca file sebagai data URL
      reader.readAsDataURL(file);
    }
  };
  const performPrediction = async () => {
    if (!model || !selectedImage) {
      console.log("Model not loaded or image not selected");
      return;
    }

    const img = new Image();
    img.src = selectedImage;

    img.onload = async () => {
      const tensor = tf.browser
        .fromPixels(img, 3) // Ambil hanya 3 channel (RGB)
        .resizeNearestNeighbor([224, 224]) // Ubah ukuran menjadi 224x224
        .toFloat()
        .div(tf.scalar(255)) // Normalisasi ke skala [0, 1]
        .expandDims(0); // Tambahkan batch dimension

      const prediction = model.predict(tensor) as tf.Tensor;

      const predictionData = await prediction.data();
      const predictedIndex = predictionData.indexOf(
        Math.max(...predictionData)
      );

      console.log(predictionData);

      // Gunakan class labels sesuai dengan model Anda
      const classLabels = [
        "Armyworms",
        "BrownMarmoratedStinkBugs",
        "CabbageLoopers",
        "CitrusCanker",
        "ColoradoPotatoBeetles",
        "CornBorers",
        "CornEarworms",
        "FallArmyworms",
        "FruitFlies",
        "Thrips",
        "Tomato_Hornworms",
      ];
      const label = classLabels[predictedIndex];
      setPredictedLabel(label);
      console.log(`Predicted Class: ${label}`);
    };
  };
  return (
    <Flex flexDirection="column" alignItems="center" gap="5" marginTop="8">
      <Heading size="2xl" textAlign="center">
        Ayo Upload Gambar
      </Heading>
      <Center>
        <FileUploadRoot onChange={handleImageUpload}>
          <FileUploadTrigger asChild>
            <Button variant="outline" size="sm">
              <HiUpload /> Upload Gambar
            </Button>
          </FileUploadTrigger>
        </FileUploadRoot>
      </Center>
      {selectedImage && (
        <Box
          p="2"
          borderWidth="1px"
          borderColor="border.disabled"
          color="fg.disabled"
          rounded="lg"
        >
          <ImageChakra
            src={selectedImage}
            aspectRatio={1 / 1}
            objectFit="cover"
            width="150px"
            alt="Dan Abramov"
          />
        </Box>
      )}

      <Button
        onClick={performPrediction}
        type="button"
        disabled={!selectedImage}
      >
        Prediksi
      </Button>
      {predictedlabel && (
        <Box
          p="2"
          borderWidth="1px"
          borderColor="border.disabled"
          color="fg.disabled"
          rounded="lg"
        >
          <Heading size="sm">
            <Flex alignItems="center" gap="2">
              Label Klasifikasi : <Tag size="lg">{predictedlabel}</Tag>
            </Flex>
          </Heading>
        </Box>
      )}
    </Flex>
  );
}

export default App;
