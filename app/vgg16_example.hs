{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Main (main) where

import Control.Applicative
import Control.Monad
import Data.List
import Data.Monoid
import Data.Ord
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import Data.Version
import qualified Graphics.Image as HIP
import Options.Applicative
import Menoh
import Text.Printf

main :: IO ()
main = do
  putStrLn "vgg16 example"
  opt <- execParser parserInfo

  let batch_size  = 1
      channel_num = 3
      height = 224
      width  = 224
      category_num = 1000
      input_dims, output_dims :: Dims
      input_dims  = [batch_size, channel_num, height, width]
      output_dims = [batch_size, category_num]

  img <- HIP.readImageRGB HIP.VS $ optInputImagePath opt
  let image_data :: VS.Vector Float
      image_data = convert width height img

  -- Aliases to onnx's node input and output tensor name
  let conv1_1_in_name  = "140326425860192"
      fc6_out_name     = "140326200777584"
      softmax_out_name = "140326200803680"

  -- Load ONNX model data
  model_data <- makeModelDataFromONNX (optModelPath opt)

  -- Specify inputs and outputs
  vpt <- makeVariableProfileTable
           [(conv1_1_in_name, DTypeFloat, input_dims)]
           [(fc6_out_name, DTypeFloat), (softmax_out_name, DTypeFloat)]
           model_data
  optimizeModelData model_data vpt

  -- Construct computation primitive list and memories
  model <- makeModel vpt model_data "mkldnn"

  -- Copy input image data to model's input array
  writeBuffer model conv1_1_in_name image_data

  -- Run inference
  run model

  -- Get output
  (fc6_out :: V.Vector Float) <- readBuffer model fc6_out_name
  putStr "fc6_out: "
  forM_ [0..4] $ \i -> do
    putStr $ show $ fc6_out V.! i
    putStr " "
  putStrLn "..."

  (softmax_out :: V.Vector Float) <- readBuffer model softmax_out_name

  categories <- liftM lines $ readFile (optSynsetWordsPath opt)
  let k = 5
  scores <- forM [0 .. V.length softmax_out - 1] $ \i -> do
    return (i, softmax_out V.! i)
  printf "top %d categories are:\n" k
  forM_ (take k $ sortBy (flip (comparing snd)) scores) $ \(i,p) -> do
    printf "%d %f %s\n" i p (categories !! i)

-- -------------------------------------------------------------------------

data Options
  = Options
  { optInputImagePath  :: FilePath
  , optModelPath       :: FilePath
  , optSynsetWordsPath :: FilePath
  }

optionsParser :: Parser Options
optionsParser = Options
  <$> inputImageOption
  <*> modelPathOption
  <*> synsetWordsPathOption
  where
    inputImageOption = strOption
      $  long "input-image"
      <> short 'i'
      <> metavar "PATH"
      <> help "input image path"
      <> value "data/Light_sussex_hen.jpg"
      <> showDefault
    modelPathOption = strOption
      $  long "model"
      <> short 'm'
      <> metavar "PATH"
      <> help "onnx model path"
      <> value "data/VGG16.onnx"
      <> showDefault
    synsetWordsPathOption = strOption
      $  long "synset-words"
      <> short 's'
      <> metavar "PATH"
      <> help "synset words path"
      <> value "data/synset_words.txt"
      <> showDefault

parserInfo :: ParserInfo Options
parserInfo = info (helper <*> versionOption <*> optionsParser)
  $  fullDesc
  <> header "vgg16_example - an example program of Menoh haskell binding"
  where
    versionOption :: Parser (a -> a)
    versionOption = infoOption (showVersion version)
      $  hidden
      <> long "version"
      <> help "Show version"

-- -------------------------------------------------------------------------

convert :: Int -> Int -> HIP.Image HIP.VS HIP.RGB Double -> VS.Vector Float
convert w h = reorderToNCHW . HIP.resize HIP.Bilinear HIP.Edge (h,w) . crop

crop :: HIP.Array arr cs e => HIP.Image arr cs e -> HIP.Image arr cs e
crop img = HIP.crop (base_y, base_x) (shortEdge, shortEdge) img
  where
    (height, width) = HIP.dims img
    shortEdge = min width height
    base_x = (width - shortEdge) `div` 2
    base_y = (height - shortEdge) `div` 2

-- Note that VGG16.onnx assumes BGR image
reorderToNCHW :: HIP.Image HIP.VS HIP.RGB Double -> VS.Vector Float
reorderToNCHW img = VS.generate (3 * height * width) f
  where
    (height, width) = HIP.dims img
    f i =
      case HIP.index img (y,x) of
        HIP.PixelRGB r g b ->
          case ch of
            0 -> realToFrac b * 255
            1 -> realToFrac g * 255
            2 -> realToFrac r * 255
            _ -> undefined
      where
        (ch,m) = i `divMod` (width * height)
        (y,x) = m `divMod` width

-- -------------------------------------------------------------------------
