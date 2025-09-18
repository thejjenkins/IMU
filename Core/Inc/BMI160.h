/****************************************************************************************************************/
/*
 ############## SENSORS ADDRESSES
*/
#define BMI160_ADDRESS 0x69
/****************************************************************************************************************/
/*
 ############# REGISTERS ##############
*/
#define MPU9150_ACCEL_XOUT        0x13 // x axis acceleration measurement
#define MPU9150_ACCEL_YOUT        0x15 // y axis acceleration measurement
#define MPU9150_ACCEL_ZOUT        0x17 // z axis acceleration measurement
#define MPU9150_GYRO_XOUT         0x0D // x axis gyroscope measurement
#define MPU9150_GYRO_YOUT         0x0F // y axis gyroscope measurement
#define MPU9150_GYRO_ZOUT         0x11 // z axis gyroscope measurement
/****************************************************************************************************************/

float getAccX(void);		 // m/s²
float getAccY(void);		 // m/s²
float getAccZ(void);		 // m/s²
float getRoll(void);		 // degrees
float getPitch(void);		 // degrees
int16_t compl_dec_convert(uint16_t data);
uint8_t SensorHub_read8(char addr);
uint16_t SensorHub_read16(char addr);
bool SensorHub_write8(uint8_t addr,uint8_t reg);
bool SensorHub_write16(uint8_t addr,uint8_t reg1,uint8_t reg2);
/****************************************************************************************************************/
