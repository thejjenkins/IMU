#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <x86intrin.h>  // for __rdtsc (on GCC/Clang/MinGW)
#include <windows.h>

#define SAMPLES_PER_AXIS 3
#define AXES 2
#define MEMORY_BLOCKS SAMPLES_PER_AXIS*AXES
#define WINDOW_LENGTH 18
#define ALPHA 0.5

typedef struct {
	// Acceleration data (x,y,z) in m/s^2
	float acc_mps2[3];

	// Gyroscope data (x,y,z) in degrees/s
	float gyro_deg[3];

}accelerometer;

typedef struct {
    float v[3]; // x = v[0], y = v[1], z = v[2]
}RollingWindow;

typedef struct {
    float out;
}FirstOrderIIR;

void FirstOrderIIR_Init(FirstOrderIIR *filt) {
    filt->out = 0.0f;
}

float FirstOrderIIR_Update(FirstOrderIIR *filt, float in) {
    filt->out = ALPHA*in + (1.0f - ALPHA)*filt->out;
    // return filt->out;
}

/**
 * @brief Updates the ring buffer with new accelerometer or gyroscope data.
 * @param data Array of 3 floats representing accelerometer or gyroscope data (x, y, z)
 * @param window Pointer to the ring buffer struct of size 3 floats
 */
void ringBuffer(float data[3], RollingWindow* window) {
    // Shift the window to make room for new data
    memmove(&window[1], &window[0], (SAMPLES_PER_AXIS - 1) * sizeof(window[0]));
    window[0].v[0] = data[0];
    window[0].v[1] = data[1];
    window[0].v[2] = data[2];
}

/**
 * @brief Finds the median of three float values without requiring sorting
 */
void medFilter(RollingWindow *window) {
    for (int axis = 0; axis < 3; ++axis) {  // 0=x, 1=y, 2=z
        float *a = &window[0].v[axis];
        float *b = &window[1].v[axis];
        float *c = &window[2].v[axis];

        if (*a > *b) {           // a > b
            if (*c > *b) {       // c > b
                if (*a > *c) {   // a > c
                    *b = *c;     // median is c
                } else {
                    *b = *a;     // median is a
                }
            }
        } else {                 // a <= b
            if (*c < *b) {       // c < b
                if (*a < *c) {   // a < c
                    *b = *c;     // median is c
                } else {
                    *b = *a;     // median is a
                }
            }
        }
    }
}

float s_window[WINDOW_LENGTH] = {
    17.0f,  3.0f, 24.0f,  9.0f, 29.0f, 
     1.0f, 21.0f, 13.0f,  6.0f, 27.0f,
    10.0f,  4.0f, 19.0f, 30.0f,  8.0f,
    25.0f,  2.0f, 10.0f
};


int main() {
    accelerometer acc = {
        .acc_mps2 = {0.5f, 1.5f, 2.5f},
        .gyro_deg = {3.5f, 4.5f, 5.5f}
    };
    FirstOrderIIR accIIR[3], gyroIIR[3];
    RollingWindow accel[3], gyro[3];
    for(int i = 0; i < 3; i++) {
        accel[i].v[0] = s_window[i*3];
        accel[i].v[1] = s_window[i*3+1];
        accel[i].v[2] = s_window[i*3+2];
        gyro[i].v[0] = s_window[i*3+9];
        gyro[i].v[1] = s_window[i*3+10];
        gyro[i].v[2] = s_window[i*3+11];
    }

    LARGE_INTEGER freq, qpc_start, qpc_end;
    QueryPerformanceFrequency(&freq);
    unsigned long long tsc_start, tsc_end;
    // Start both timers
    QueryPerformanceCounter(&qpc_start);
    tsc_start = __rdtsc();

    for(int i=0; i<3; i++){
        FirstOrderIIR_Init(&accIIR[i]);
        FirstOrderIIR_Init(&gyroIIR[i]);
    }
    for (int i = 0; i < SAMPLES_PER_AXIS; i++) {
        printf("accel[%d].v[0] = %.3f\n", i, accel[i].v[0]);
        printf("accel[%d].v[1] = %.3f\n", i, accel[i].v[1]);
        printf("accel[%d].v[2] = %.3f\n", i, accel[i].v[2]);
    }
    printf("***************\n");
    ringBuffer(acc.acc_mps2, &accel[0]);
    ringBuffer(acc.gyro_deg, &gyro[0]);
    for (int i = 0; i < SAMPLES_PER_AXIS; i++) {
        printf("accel[%d].v[0] = %.3f\n", i, accel[i].v[0]);
        printf("accel[%d].v[1] = %.3f\n", i, accel[i].v[1]);
        printf("accel[%d].v[2] = %.3f\n", i, accel[i].v[2]);
    }
    printf("***************\n");
    medFilter(&accel[0]);
    medFilter(&gyro[0]);
    for (int i = 0; i < SAMPLES_PER_AXIS; i++) {
        printf("accel[%d].v[0] = %.3f\n", i, accel[i].v[0]);
        printf("accel[%d].v[1] = %.3f\n", i, accel[i].v[1]);
        printf("accel[%d].v[2] = %.3f\n", i, accel[i].v[2]);
    }
    printf("***************\n");
    for (int axis = 0; axis < 3; ++axis) {
        FirstOrderIIR_Update(&accIIR[axis], accel[1].v[axis]);
        FirstOrderIIR_Update(&gyroIIR[axis], gyro[1].v[axis]);
        printf("Filtered acc axis %d: %.3f\n", axis, accIIR[axis].out);
        printf("Filtered gyro axis %d: %.3f\n", axis, gyroIIR[axis].out);
    }


    tsc_end = __rdtsc();
    QueryPerformanceCounter(&qpc_end);

    // Convert QPC delta to seconds
    long long qpc_delta = qpc_end.QuadPart - qpc_start.QuadPart;
    double seconds = (double)qpc_delta / (double)freq.QuadPart;

    // TSC delta
    unsigned long long tsc_delta = tsc_end - tsc_start;

    // Now compute TSC frequency
    double tsc_hz = (double)tsc_delta / seconds;
    double tsc_ghz = tsc_hz / 1e9;

    printf("TSC delta: %llu cycles\n", tsc_delta);
    printf("Elapsed: %.8f seconds\n", seconds);
    printf("Estimated TSC frequency: %.3f GHz\n", tsc_ghz);
    
    return 0;
}