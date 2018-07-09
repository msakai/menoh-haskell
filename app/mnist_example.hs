{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Main (main) where

import Control.Applicative
import Control.Monad
import Data.Monoid
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import Data.Version
import qualified Graphics.Image as HIP
import qualified Graphics.Image.Interface as HIP
import Options.Applicative
import Menoh
import System.FilePath
import Text.Printf

import Paths_menoh (getDataDir)

main :: IO ()
main = do
  putStrLn "mnist example"
  dataDir <- getDataDir
  opt <- execParser (parserInfo (dataDir </> "data"))

  let input_dir = optInputPath opt
      image_filenames =
        [ "0.png"
        , "1.png"
        , "2.png"
        , "3.png"
        , "4.png"
        , "5.png"
        , "6.png"
        , "7.png"
        , "8.png"
        , "9.png"
        ]
      batch_size  = length image_filenames
      channel_num = 1
      height = 28
      width  = 28
      category_num = 10
      input_dims, output_dims :: Dims
      input_dims  = [batch_size, channel_num, height, width]
      output_dims = [batch_size, category_num]

  images <- liftM VS.concat $ forM image_filenames $ \fname -> do
    img <- HIP.readImageY HIP.VS $ input_dir </> fname
    return
      $ VS.map (\(HIP.PixelY y) -> realToFrac y * 255 :: Float)
      $ HIP.toVector $ HIP.resize HIP.Bilinear HIP.Edge (height,width)
      $ img

  -- Aliases to onnx's node input and output tensor name
  let mnist_in_name  = "139900320569040"
      mnist_out_name = "139898462888656"

  -- Load ONNX model data
  model_data <- makeModelDataFromONNX (optModelPath opt)

  -- Specify inputs and outputs
  vpt <- makeVariableProfileTable
           [(mnist_in_name, DTypeFloat, input_dims)]
           [(mnist_out_name, DTypeFloat)]
           model_data
  optimizeModelData model_data vpt

  -- Construct computation primitive list and memories
  model <- makeModel vpt model_data "mkldnn"

  -- Copy input image data to model's input array
  writeBuffer model mnist_in_name images

  -- Run inference
  run model

  -- Get output
  (v :: V.Vector Float) <- readBuffer model mnist_out_name
  forM_ (zip [0..] image_filenames) $ \(i,fname) -> do
    let scores = V.slice (i * category_num) category_num v
        j = V.maxIndex scores
        s = scores V.! j
    printf "%s = %d : %f\n" fname j s

-- -------------------------------------------------------------------------

data Options
  = Options
  { optInputPath :: FilePath
  , optModelPath :: FilePath
  }

optionsParser :: FilePath -> Parser Options
optionsParser dataDir = Options
  <$> inputPathOption
  <*> modelPathOption
  where
    inputPathOption = strOption
      $  long "input"
      <> short 'i'
      <> metavar "DIR"
      <> help "input image path"
      <> value dataDir
      <> showDefault
    modelPathOption = strOption
      $  long "model"
      <> short 'm'
      <> metavar "PATH"
      <> help "onnx model path"
      <> value (dataDir </> "mnist.onnx")
      <> showDefault

parserInfo :: FilePath -> ParserInfo Options
parserInfo dir = info (helper <*> versionOption <*> optionsParser dir)
  $  fullDesc
  <> header "mnist_example - an example program of Menoh haskell binding"
  where
    versionOption :: Parser (a -> a)
    versionOption = infoOption (showVersion version)
      $  hidden
      <> long "version"
      <> help "Show version"

-- -------------------------------------------------------------------------
