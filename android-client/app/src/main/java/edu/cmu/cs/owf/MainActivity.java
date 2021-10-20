package edu.cmu.cs.owf;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.ImageView;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import java.util.Locale;
import java.util.function.Consumer;

import edu.cmu.cs.gabriel.camera.CameraCapture;
import edu.cmu.cs.gabriel.camera.ImageViewUpdater;
import edu.cmu.cs.gabriel.camera.YuvToJPEGConverter;
import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.protocol.Protos;
import edu.cmu.cs.owf.Protos.ToClientExtras;
import edu.cmu.cs.owf.Protos.ToServerExtras;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final String SOURCE = "owf_client";
    private static final int PORT = 9099;

    // Attempt to get the largest images possible. ImageAnalysis is limited to something below 1080p
    // according to this:
    // https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Builder#setTargetResolution(android.util.Size)
    private static final int WIDTH = 1920;
    private static final int HEIGHT = 1080;

    private String step;

    private ServerComm serverComm;
    private YuvToJPEGConverter yuvToJPEGConverter;
    private CameraCapture cameraCapture;

    private TextToSpeech textToSpeech;
    private ImageViewUpdater cropViewUpdater;
    private ImageViewUpdater instructionViewUpdater;

    private final Consumer<Protos.ResultWrapper> consumer = resultWrapper -> {
        try {
            ToClientExtras toClientExtras = ToClientExtras.parseFrom(
                    resultWrapper.getExtras().getValue());
            step = toClientExtras.getStep();

        } catch (InvalidProtocolBufferException e) {
            Log.e(TAG, "Protobuf parse error", e);
        }

        if (resultWrapper.getResultsCount() == 0) {
            return;
        }

        for (Protos.ResultWrapper.Result result : resultWrapper.getResultsList()) {
            if (result.getPayloadType() == Protos.PayloadType.TEXT) {
                ByteString dataString = result.getPayload();
                String speech = dataString.toStringUtf8();
                this.textToSpeech.speak(speech, TextToSpeech.QUEUE_ADD, null, null);
            } else {
                ByteString image = result.getPayload();
                instructionViewUpdater.accept(image);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        PreviewView viewFinder = findViewById(R.id.viewFinder);
        ImageView cropView = findViewById(R.id.cropView);
        cropViewUpdater = new ImageViewUpdater(cropView);
        ImageView instructionView = findViewById(R.id.instructionView);
        instructionViewUpdater = new ImageViewUpdater(instructionView);

        Consumer<ErrorType> onDisconnect = errorType -> {
            Log.e("MainActivity", "Disconnect Error: " + errorType.name());
            finish();
        };
        serverComm = ServerComm.createServerComm(
                consumer, BuildConfig.GABRIEL_HOST, PORT, getApplication(), onDisconnect);

        TextToSpeech.OnInitListener onInitListener = i -> {
            textToSpeech.setLanguage(Locale.US);

            ToServerExtras toServerExtras = ToServerExtras.newBuilder().setStep("").build();
            Protos.InputFrame inputFrame = Protos.InputFrame.newBuilder()
                    .setExtras(pack(toServerExtras))
                    .build();

            // We need to wait for textToSpeech to be initialized before asking for the first
            // instruction.
            serverComm.send(inputFrame, SOURCE, /* wait */ true);
        };
        this.textToSpeech = new TextToSpeech(this, onInitListener);

        yuvToJPEGConverter = new YuvToJPEGConverter(this, 100);
        cameraCapture = new CameraCapture(this, analyzer, WIDTH, HEIGHT, viewFinder, CameraSelector.DEFAULT_BACK_CAMERA, false);
    }

    // Based on
    // https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/compiler/java/java_message.cc#L1387
    public static Any pack(ToServerExtras toServerExtras) {
        return Any.newBuilder()
                .setTypeUrl("type.googleapis.com/owf.ToServerExtras")
                .setValue(toServerExtras.toByteString())
                .build();
    }

    final private ImageAnalysis.Analyzer analyzer = new ImageAnalysis.Analyzer() {
        @Override
        public void analyze(@NonNull ImageProxy image) {
            serverComm.sendSupplier(() -> {
                ByteString jpegByteString = yuvToJPEGConverter.convert(image);

                ToServerExtras toServerExtras = ToServerExtras.newBuilder()
                        .setStep(MainActivity.this.step)
                        .build();

                return Protos.InputFrame.newBuilder()
                        .setPayloadType(Protos.PayloadType.IMAGE)
                        .addPayloads(jpegByteString)
                        .setExtras(pack(toServerExtras))
                        .build();
            }, SOURCE, /* wait */ false);

            // The image has either been sent or skipped. It is therefore safe to close the image.
            image.close();
        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraCapture.shutdown();
    }
}