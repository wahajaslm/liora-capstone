# Challenge context

## Rakuten

Rakuten, created in 1997 in Japan and at the origin of the marketplace concept, became one of the largest e-commerce platforms worldwide with a community of more than 1.3 billion members. Along with the global marketplaces, Rakuten supports an ever-expanding list of acquisitions and strategic investments in disruptive industries and growing markets, such as communications, financial services, digital contents, and gathers more than one billion users in an international ecosystem.

Rakuten Institute of Technology (RIT) is the research and innovation department of Rakuten, with teams in Tokyo, Paris, Boston, Singapore, Bengaluru. RIT does applied research in the domains of computer vision, natural language processing, machine / deep Learning and human-computer interaction.

## Context

This challenge focuses on the topic of large-scale product type code multimodal (text and image) classification where the goal is to predict each product’s type code as defined in the catalog of Rakuten France.

The cataloging of product listings through title and image categorization is a fundamental problem for any e-commerce marketplace, with applications ranging from personalized search and recommendations to query understanding. Manual and rule-based approaches to categorization are not scalable since commercial products are organized in many classes. Deploying multimodal approaches would be a useful technique for e-commerce companies as they have trouble categorizing products given images and labels from merchants and avoid duplication, especially when selling both new and used products from professional and non-professional merchants, like Rakuten does. Advances in this area of research have been limited due to the lack of real data from actual commercial catalogs. The challenge presents several interesting research aspects due to the intrinsic noisy nature of the product labels and images, the size of modern e-commerce catalogs, and the typical unbalanced data distribution.

---

# Challenge goals

## Problem Description

The goal of this data challenge is large-scale multimodal (text and image) product data classification into product type codes.

For example, in Rakuten France catalog, a product with a French designation or title *Klarstein Présentoir 2 Montres Optique Fibre* associated with an image and sometimes with an additional description. This product is categorized under the 1500 product type code. There are other products with different titles, images and with possible descriptions, which are under the same product type code. Given these information on the products, like the example above, this challenge proposes to model a classifier to classify the products into its corresponding product type code.

## Metric

The metric used in this challenge to rank the participants is the weighted-F1 score.

Scikit-Learn package has an F1 score implementation ([link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)) and can be used for this challenge with its average parameter set to `"weighted"`.

## Legal Notice

By express derogation from of Etalab Open Licence terms, the Participant acknowledges that the Study Data uploaded by the Provider on the occasion of the ENS Data Challenge is strictly confidential information. The Participant shall (i) hold in strict confidence and not disclose to any third party all or part of the Study Data, (ii) use the Study Data for the sole purpose of the good performance of the ENS Data Challenge (the “Purpose”), (iii) not use, apply, reveal, report, publish or otherwise disclose all or part of the Study Data for no purpose other the Purpose, excluding notably commercial or technical use of any kind. As of the termination of the ENS Data Challenge, the Participant shall immediately cease any use of the Study Data unless otherwise agreed by the Provider. The present specific terms shall remain in full force and effect until the termination of the Purpose and for a period of two (2) years following the termination date of the Purpose.

## Contact

For any technical questions about this challenge please contact to any of the following addresses:

- [parantapa.goswami@rakuten.com](mailto:parantapa.goswami@rakuten.com)
- [hesam.amoualian@rakuten.com](mailto:hesam.amoualian@rakuten.com)

---

# Data description

## Data Description

For this challenge, Rakuten France is releasing approx. 99K product listings in CSV format, including the train (84,916) and test set (13,812). The dataset consists of product designations, product descriptions, product images and their corresponding product type code.

The data are divided under two criteria, forming four distinct sets: training or test, input or output.

- `X_train.csv`: training input file
- `Y_train.csv`: training output file
- `X_test.csv`: test input file

Additionally `images.zip` file is supplied containing all the images. Uncompressing this file will provide a folder named `images` with two subfolders named `image_training` and `image_test`, containing training and test images respectively.

The first line of the input files contains the header, and the columns are separated by comma (`","`). The columns are:

- An integer ID for the product. This ID is used to associate the product with its corresponding product type code.
- `designation` - The product title, a short text summarizing the product.
- `description` - A more detailed text describing the product. Not all the merchants use this field, so to retain originality of the data, the description field can contain NaN value for many products.
- `productid` - An unique ID for the product.
- `imageid` - An unique ID for the image associated with the product.

The fields `imageid` and `productid` are used to retrieve the images from the respective image folder. For a particular product the image file name is `image_imageid_product_productid.jpg`.

Here is an example of an input file:

```csv
,designation,description,productid,imageid
0,Olivia: Personalisiertes Notizbuch 150 Seiten Punktraster Ca Din A5 Rosen-Design,,3804725264,1263597046
1,Journal Des Arts (Le) N° 133 Du 28/09/2001 - L'art Et Son Marche Salon D'art Asiatique A Paris - Jacques Barrere - Francois Perrier - La Reforme Des Ventes Aux Encheres Publiques - Le Sna Fete Ses Cent Ans.,,436067568,1008141237
```

For the first product the corresponding image file name is `image_1263597046_product_3804725264.jpg`, and the same for the second product is `image_1008141237_product_436067568.jpg`. One can recall that all the images corresponding to the training products listed in `X_train.csv` can be found in `image_training` subfolder, and all the images corresponding to the test products listed in `X_test.csv` can be found in `image_test` subfolder.

The training output file (`Y_train.csv`) contains the `prdtypecode`, the category for the classification task, for each integer id in the training input file (`X_train.csv`). Here also the first line of the file is the header and columns are separated by commas.

Here is an example of the output file:

```csv
,prdtypecode
0,10
1,2280
```

For the test input file `X_test.csv`, participants need to provide a test output file in the same format as the training output file (associating each integer id with the predicted `prdtypecode`). The first line of this test output file should contain the header `,prdtypecode`.

---

# Benchmark description

## Benchmark Model

The benchmark algorithm uses two separate models for the images and the text. Participants can get an idea of the performances when these sources of informations are used separately. They are encouraged to use both these sources while designing a classifier, since they contain complementary information.

For the image data, a version of Residual Networks (ResNet) model (reference) is used. ResNet50 implementation from Keras is used as the base model. The details of the basic benchmark model can be found in this notebook. The model is a pre-trained ResNet50 with ImageNet dataset. 27 different layers from top are unfrozen, which include 8 Convolutional layers for the training. The final network contains 12,144,667 trainable and 23,643,035 non-trainable parameters.

For the text data a simplified CNN classifier used. Only the designation fields (product titles) are used in this benchmark model. The input size is the maximum possible designation length, 34 in this case. Shorter inputs are zero-padded. The architecture consists of an embedding layer and 6 convolutional, max-pooling blocks. The embeddings are trained with the entire architecture. Following is the model architecture:

| Layer (type) | Output Shape | Number of Params | Connected to |
| --- | --- | --- | --- |
| InputLayer | (None, 34) | 0 | |
| Embedding Layer | (None, 34, 300) | 17320500 | InputLayer |
| Reshape | (None, 34, 300, 1) | 0 | Embedding Layer |
| Conv2D Block 1 | (None, 34, 1, 512) | 154112 | Reshape |
| MaxPooling2D Block 1 | (None, 1, 1, 512) | 0 | Conv2D Block 1 |
| Conv2D Block 2 | (None, 33, 1, 512) | 307712 | Reshape |
| MaxPooling2D Block 2 | (None, 1, 1, 512) | 0 | Conv2D Block 2 |
| Conv2D Block 3 | (None, 32, 1, 512) | 461312 | Reshape |
| MaxPooling2D Block 3 | (None, 1, 1, 512) | 0 | Conv2D Block 2 |
| Conv2D Block 4 | (None, 31, 1, 512) | 614912 | Reshape |
| MaxPooling2D Block 4 | (None, 1, 1, 512) | 0 | Conv2D Block 2 |
| Conv2D Block 5 | (None, 30, 1, 512) | 768512 | Reshape |
| MaxPooling2D Block 5 | (None, 1, 1, 512) | 0 | Conv2D Block 2 |
| Conv2D Block 6 | (None, 29, 1, 512) | 922112 | Reshape |
| MaxPooling2D Block 6 | (None, 1, 1, 512) | 0 | Conv2D Block 2 |
| Concatenate | (None, 6, 1, 512) | 0 | All MaxPooling2D Blocks |
| Flatten | (None, 3072) | 0 | Concatenate |
| Dropout Layer | (None, 3072) | 0 | Flatten |
| Dense Layer | (None, 27) | 8297 | Dropout Layer |

This architecture contains total 20,632,143 trainable parameters.

## Benchmark Performance

Following are the weighted-F1 score obtained using the benchmark models described above:

- **Text:** 0.8113
- **Images:** 0.5534

As the benchmarking model using text is better performing, the Y benchmark file contains the output of the same.