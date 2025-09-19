#include "BMI160.h"

uint8_t BMI160_Initialize( BMI160 *dev, I2C_HandleTypeDef *i2cHandle ) {
	// Set struct parameters
	dev->i2cHandle = i2cHandle;

	dev->acc_mps2[0] = 0.0f;
	dev->acc_mps2[1] = 0.0f;
	dev->acc_mps2[2] = 0.0f;

	dev->gyro_deg[0] = 0.0f;
	dev->gyro_deg[1] = 0.0f;
	dev->gyro_deg[2] = 0.0f;

	// Store number of transaction errors (to be returned at end of function)
	uint8_t errNum = 0;
	HAL_StatusTypeDef status;

	/*
	 * Check device ID (datasheet pg. 49)
	 */
	uint8_t regData;

	status = BMI160_ReadRegister( dev, BMI160_REG_CHIP_ID, &regData );
	errNum += ( status != HAL_OK );

	if ( regData != BMI160_CHIP_ID ) {
		return 255;
	}

	/*
	 * Power Management Unit (PMU) Configuration (p. 13-17, 85-86)
	 * Default is suspend mode (at reset)
	 * Set PMU mode of accelerometer and gyroscope to normal
	 * Gyroscope data can only be processed in normal power mode (p. 21)
	 */
	regData = 0x14;
	status = BMI160_WriteRegister( dev, BMI160_REG_CMD, &regData );
	errNum += ( status != HAL_OK );

	/*
	 * Accelerometer Configuration (p. 19)
	 * BMI160_REG_ACC_CONF (0x40) default value is 0x28 (p. 49, 58)
	 * Default configuration supports 3-dB cutoff frequency of 40.5 Hz
	 * and ODR of 100 Hz
	 * BMI160_REG_ACC_RANGE (0x41) default value is 0x03 (p. 48, 59)
	 * Default configuration supports +-2g range
	 */
	status = BMI160_ReadRegister( dev, BMI160_REG_ACC_CONF, &regData );
	errNum += ( status != HAL_OK );
	status = BMI160_ReadRegister( dev, BMI160_REG_ACC_RANGE, &regData );
	errNum += ( status != HAL_OK );

	/*
	 * Gyroscope Configuration (p. 21)
	 * BMI160_REG_GYRO_CONF (0x42) default value is 0x28 (p. 48, 60)
	 * Default configuration supports 3-dB cutoff frequency of 39.9 Hz
	 * and ODR of 100 Hz
	 * BMI160_REG_GYRO_RANGE (0x43) default value is 0x00 (p. 48, 61)
	 * Default configuration supports +-2000 deg/s
	 */
	status = BMI160_ReadRegister( dev, BMI160_REG_GYRO_CONF, &regData );
	errNum += ( status != HAL_OK );
	status = BMI160_ReadRegister( dev, BMI160_REG_GYRO_RANGE, &regData );
	errNum += ( status != HAL_OK );

	// if errNum is 0 then initialization was successful
	return errNum;
}

/*
 * DATA ACQUISITION
 */
static float BMI160_ACC_LSB_PER_G( uint8_t acc_range_reg )
{
    // acc_range<3:0>: 0b0011=±2g, 0b0101=±4g, 0b1000=±8g, 0b1100=±16g
    switch ( acc_range_reg & 0x0F ) {
        case 0x03: return 16384.0f; // ±2g
        case 0x05: return  8192.0f; // ±4g
        case 0x08: return  4096.0f; // ±8g
        case 0x0C: return  2048.0f; // ±16g
        default:   return 16384.0f; // fallback to ±2g
    }
}

static float BMI160_GYR_LSB_PER_DPS(uint8_t gyr_range_reg)
{
    // gyr_range<2:0>: 000=±2000, 001=±1000, 010=±500, 011=±250, 100=±125
    switch (gyr_range_reg & 0x07) {
        case 0x00: return  16.4f;   // ±2000 °/s
        case 0x01: return  32.8f;   // ±1000 °/s
        case 0x02: return  65.6f;   // ±500  °/s
        case 0x03: return 131.2f;   // ±250  °/s
        case 0x04: return 262.4f;   // ±125  °/s
        default:   return  16.4f;   // safe default
    }
}

// convert little-endian and return a single 16-bit signed number
static inline int16_t le16_to_s16(uint8_t lo, uint8_t hi)
{
    return (int16_t)((uint16_t)lo | ((uint16_t)hi << 8));
}

HAL_StatusTypeDef BMI160_ReadAccelerations( BMI160 *dev )
{
    uint8_t buf[6];
    HAL_StatusTypeDef st = BMI160_ReadRegisters(dev, BMI160_REG_ACC_X_7_0, buf, sizeof(buf));
    if (st != HAL_OK) return st;

    // read current acc range (for correct scaling)
    uint8_t acc_range = 0;
    st = BMI160_ReadRegister(dev, BMI160_REG_ACC_RANGE, &acc_range);
    if (st != HAL_OK) return st;

    const float lsb_per_g = BMI160_ACC_LSB_PER_G(acc_range);
    const float g_to_mps2  = 9.80665f;

    int16_t raw_x = le16_to_s16(buf[0], buf[1]);
    int16_t raw_y = le16_to_s16(buf[2], buf[3]);
    int16_t raw_z = le16_to_s16(buf[4], buf[5]);

    dev->acc_mps2[0] = (raw_x / lsb_per_g) * g_to_mps2;
    dev->acc_mps2[1] = (raw_y / lsb_per_g) * g_to_mps2;
    dev->acc_mps2[2] = (raw_z / lsb_per_g) * g_to_mps2;
    return HAL_OK;
}

HAL_StatusTypeDef BMI160_ReadGyro(BMI160 *dev)
{
    uint8_t buf[6];
    HAL_StatusTypeDef st = BMI160_ReadRegisters(dev, BMI160_REG_GYR_X_7_0, buf, sizeof(buf));
    if (st != HAL_OK) return st;

    // read current gyro range (for correct scaling)
    uint8_t gyr_range = 0;
    st = BMI160_ReadRegister(dev, BMI160_REG_GYR_RANGE, &gyr_range);
    if (st != HAL_OK) return st;

    const float lsb_per_dps = BMI160_GYR_LSB_PER_DPS(gyr_range);

    int16_t raw_x = le16_to_s16(buf[0], buf[1]);
    int16_t raw_y = le16_to_s16(buf[2], buf[3]);
    int16_t raw_z = le16_to_s16(buf[4], buf[5]);

    dev->gyro_deg[0] = raw_x / lsb_per_dps;
    dev->gyro_deg[1] = raw_y / lsb_per_dps;
    dev->gyro_deg[2] = raw_z / lsb_per_dps;
    return HAL_OK;
}

/*
 * LOW-LEVEL FUNCTIONS
 */
HAL_StatusTypeDef BMI160_ReadRegister( BMI160 *dev, uint8_t reg, uint8_t *data ) {
	return HAL_I2C_Mem_Read( dev->i2cHandle, BMI160_ADDRESS, reg, I2C_MEMADD_SIZE_8BIT, data, 1, HAL_MAX_DELAY );
}
HAL_StatusTypeDef BMI160_ReadRegisters( BMI160 *dev, uint8_t reg, uint8_t *data, uint8_t length ) {
	return HAL_I2C_Mem_Read( dev->i2cHandle, BMI160_ADDRESS, reg, I2C_MEMADD_SIZE_8BIT, data, length, HAL_MAX_DELAY);
}

HAL_StatusTypeDef BMI160_WriteRegister( BMI160 *dev, uint8_t reg, uint8_t *data ) {
	return HAL_I2C_Mem_Write(dev->i2cHandle, BMI160_ADDRESS, reg, I2C_MEMADD_SIZE_8BIT, data, 1, HAL_MAX_DELAY );
}
