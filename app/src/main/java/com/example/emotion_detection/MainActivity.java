package com.example.emotion_detection;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.emotion_detection.ml.MobModel;
import com.example.emotion_detection.ml.TfLiteModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

Button capture, predict;
ImageView camImg;
TextView tag;
private final int Camera_req_code = 100;

Intent globalData;

int imageSize = 244;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        capture = findViewById(R.id.capture);
        predict = findViewById(R.id.predict);
        camImg = findViewById(R.id.cam_image);
        tag = findViewById(R.id.predictedTag);




        capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, Camera_req_code);
                    }
                }
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, Camera_req_code);
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Bitmap img = (Bitmap) (globalData.getExtras().get("data"));
                Bitmap image = Bitmap.createScaledBitmap(img, imageSize, imageSize, false);
                classifyImage(image);
            }
        });

    }
    public void classifyImage(Bitmap image){

        try {
            int width = image.getWidth();
            int height = image.getHeight();
            Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

            MobModel model = MobModel.newInstance(getApplicationContext());

            // Creates inputs for reference.
            ByteBuffer byteBuffer =  ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0 , image.getWidth(), image.getHeight());
            int pixel = 0 ;
            for(int i=0; i<imageSize ; i++){
                for(int j=0; j < imageSize; j++){
                    int val = intValues[pixel++];

                    byteBuffer.putFloat(((val>>16) & 0xFF)  * (1.f / 255.f));
                    byteBuffer.putFloat(((val>>8) & 0xFF)  * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF)  * (1.f / 255.f));
                }
            }
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 244, 244, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            MobModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidence = outputFeature0.getFloatArray();
            Log.i("Confidence",confidence.toString() + "Confidence Length " + confidence.length);

            int maxPos = 0;
            float maxConfidence = 0;
            for(int i=0; i<confidence.length; i++)
            {
                if (confidence[i] > maxConfidence)
                {
                    maxConfidence = confidence[i];
                    maxPos = i;
                }
            }
            String [] classes = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"};
            tag.setText(classes[maxPos]);

//            System.out.println(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(this,"not predicted", Toast.LENGTH_SHORT).show();

        }


    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode==RESULT_OK)
        {
            if (requestCode==Camera_req_code)
            {
                Bitmap img = (Bitmap) (data.getExtras().get("data"));
                camImg.setImageBitmap(img);
                globalData = data;
            }
        }
    }
}