package main

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"crypto/md5"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// CompressionAlgorithm представляет тип алгоритма сжатия
type CompressionAlgorithm int

const (
	GZIP CompressionAlgorithm = iota
	ZLIB
)

// CompressionStats содержит статистику сжатия
type CompressionStats struct {
	OriginalSize   int64
	CompressedSize int64
	CompressionRatio float64
	FilesProcessed int
	ProcessingTime time.Duration
}

// ImageCompressor основная структура для сжатия изображений
type ImageCompressor struct {
	Algorithm CompressionAlgorithm
	OutputDir string
	Stats     CompressionStats
	mutex     sync.Mutex
}

// NewImageCompressor создает новый экземпляр компрессора
func NewImageCompressor(algorithm CompressionAlgorithm, outputDir string) *ImageCompressor {
	return &ImageCompressor{
		Algorithm: algorithm,
		OutputDir: outputDir,
		Stats:     CompressionStats{},
	}
}

// CompressFile сжимает отдельный файл изображения
func (ic *ImageCompressor) CompressFile(inputPath string) error {
	// Читаем исходный файл
	data, err := ioutil.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("ошибка чтения файла %s: %v", inputPath, err)
	}

	// Получаем размер исходного файла
	originalSize := int64(len(data))

	// Сжимаем данные
	compressedData, err := ic.compressData(data)
	if err != nil {
		return fmt.Errorf("ошибка сжатия файла %s: %v", inputPath, err)
	}

	// Создаем выходной путь
	outputPath := ic.generateOutputPath(inputPath)
	
	// Создаем директорию если не существует
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("ошибка создания директории: %v", err)
	}

	// Записываем сжатые данные
	if err := ioutil.WriteFile(outputPath, compressedData, 0644); err != nil {
		return fmt.Errorf("ошибка записи сжатого файла: %v", err)
	}

	// Обновляем статистику
	ic.updateStats(originalSize, int64(len(compressedData)))

	fmt.Printf("Файл сжат: %s -> %s (%.2f%% от исходного размера)\n", 
		inputPath, outputPath, float64(len(compressedData))/float64(originalSize)*100)

	return nil
}

// compressData выполняет сжатие данных выбранным алгоритмом
func (ic *ImageCompressor) compressData(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	
	switch ic.Algorithm {
	case GZIP:
		writer := gzip.NewWriter(&buf)
		if _, err := writer.Write(data); err != nil {
			return nil, err
		}
		if err := writer.Close(); err != nil {
			return nil, err
		}
	case ZLIB:
		writer := zlib.NewWriter(&buf)
		if _, err := writer.Write(data); err != nil {
			return nil, err
		}
		if err := writer.Close(); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("неподдерживаемый алгоритм сжатия")
	}
	
	return buf.Bytes(), nil
}

// DecompressFile распаковывает сжатый файл
func (ic *ImageCompressor) DecompressFile(inputPath string) error {
	// Читаем сжатый файл
	data, err := ioutil.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("ошибка чтения сжатого файла: %v", err)
	}

	// Распаковываем данные
	decompressedData, err := ic.decompressData(data)
	if err != nil {
		return fmt.Errorf("ошибка распаковки: %v", err)
	}

	// Создаем выходной путь
	outputPath := strings.TrimSuffix(inputPath, filepath.Ext(inputPath))
	
	// Записываем распакованные данные
	if err := ioutil.WriteFile(outputPath, decompressedData, 0644); err != nil {
		return fmt.Errorf("ошибка записи распакованного файла: %v", err)
	}

	fmt.Printf("Файл распакован: %s -> %s\n", inputPath, outputPath)
	return nil
}

// decompressData выполняет распаковку данных
func (ic *ImageCompressor) decompressData(data []byte) ([]byte, error) {
	buf := bytes.NewReader(data)
	
	switch ic.Algorithm {
	case GZIP:
		reader, err := gzip.NewReader(buf)
		if err != nil {
			return nil, err
		}
		defer reader.Close()
		return ioutil.ReadAll(reader)
	case ZLIB:
		reader, err := zlib.NewReader(buf)
		if err != nil {
			return nil, err
		}
		defer reader.Close()
		return ioutil.ReadAll(reader)
	default:
		return nil, fmt.Errorf("неподдерживаемый алгоритм сжатия")
	}
}

// ProcessDirectory обрабатывает все изображения в директории
func (ic *ImageCompressor) ProcessDirectory(inputDir string) error {
	startTime := time.Now()
	
	err := filepath.Walk(inputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		if !info.IsDir() && ic.isImageFile(path) {
			if err := ic.CompressFile(path); err != nil {
				log.Printf("Ошибка обработки файла %s: %v", path, err)
				return nil // Продолжаем обработку других файлов
			}
		}
		return nil
	})
	
	ic.Stats.ProcessingTime = time.Since(startTime)
	return err
}

// ProcessBatch обрабатывает список файлов параллельно
func (ic *ImageCompressor) ProcessBatch(filePaths []string, workers int) error {
	startTime := time.Now()
	
	jobs := make(chan string, len(filePaths))
	results := make(chan error, len(filePaths))
	
	// Запускаем воркеры
	for w := 0; w < workers; w++ {
		go func() {
			for filePath := range jobs {
				results <- ic.CompressFile(filePath)
			}
		}()
	}
	
	// Отправляем задания
	for _, filePath := range filePaths {
		jobs <- filePath
	}
	close(jobs)
	
	// Собираем результаты
	var errors []string
	for i := 0; i < len(filePaths); i++ {
		if err := <-results; err != nil {
			errors = append(errors, err.Error())
		}
	}
	
	ic.Stats.ProcessingTime = time.Since(startTime)
	
	if len(errors) > 0 {
		return fmt.Errorf("ошибки при обработке файлов: %s", strings.Join(errors, "; "))
	}
	
	return nil
}

// generateOutputPath создает путь для выходного файла
func (ic *ImageCompressor) generateOutputPath(inputPath string) string {
	filename := filepath.Base(inputPath)
	ext := ""
	
	switch ic.Algorithm {
	case GZIP:
		ext = ".gz"
	case ZLIB:
		ext = ".zlib"
	}
	
	return filepath.Join(ic.OutputDir, filename+ext)
}

// isImageFile проверяет, является ли файл изображением
func (ic *ImageCompressor) isImageFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	imageExts := []string{".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}
	
	for _, imgExt := range imageExts {
		if ext == imgExt {
			return true
		}
	}
	return false
}

// updateStats обновляет статистику сжатия
func (ic *ImageCompressor) updateStats(originalSize, compressedSize int64) {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	
	ic.Stats.OriginalSize += originalSize
	ic.Stats.CompressedSize += compressedSize
	ic.Stats.FilesProcessed++
	
	if ic.Stats.OriginalSize > 0 {
		ic.Stats.CompressionRatio = float64(ic.Stats.CompressedSize) / float64(ic.Stats.OriginalSize)
	}
}

// GetStats возвращает статистику сжатия
func (ic *ImageCompressor) GetStats() CompressionStats {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	return ic.Stats
}

// PrintStats выводит статистику в удобном формате
func (ic *ImageCompressor) PrintStats() {
	stats := ic.GetStats()
	fmt.Printf("\n=== СТАТИСТИКА СЖАТИЯ ===\n")
	fmt.Printf("Обработано файлов: %d\n", stats.FilesProcessed)
	fmt.Printf("Исходный размер: %.2f МБ\n", float64(stats.OriginalSize)/(1024*1024))
	fmt.Printf("Сжатый размер: %.2f МБ\n", float64(stats.CompressedSize)/(1024*1024))
	fmt.Printf("Коэффициент сжатия: %.3f\n", stats.CompressionRatio)
	fmt.Printf("Экономия места: %.1f%%\n", (1-stats.CompressionRatio)*100)
	fmt.Printf("Время обработки: %v\n", stats.ProcessingTime)
	
	if stats.ProcessingTime.Seconds() > 0 {
		rate := float64(stats.FilesProcessed) / stats.ProcessingTime.Seconds()
		fmt.Printf("Скорость обработки: %.1f файлов/сек\n", rate)
	}
}

// CalculateFileHash вычисляет MD5 хеш файла для проверки целостности
func CalculateFileHash(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	hasher := md5.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}
	
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

// VerifyCompression проверяет целостность сжатого файла
func (ic *ImageCompressor) VerifyCompression(originalPath, compressedPath string) error {
	// Вычисляем хеш исходного файла
	originalHash, err := CalculateFileHash(originalPath)
	if err != nil {
		return fmt.Errorf("ошибка вычисления хеша исходного файла: %v", err)
	}
	
	// Распаковываем временно и вычисляем хеш
	tempDir, err := ioutil.TempDir("", "verify_")
	if err != nil {
		return fmt.Errorf("ошибка создания временной директории: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	tempFile := filepath.Join(tempDir, "temp_decompressed")
	
	// Читаем и распаковываем сжатый файл
	compressedData, err := ioutil.ReadFile(compressedPath)
	if err != nil {
		return fmt.Errorf("ошибка чтения сжатого файла: %v", err)
	}
	
	decompressedData, err := ic.decompressData(compressedData)
	if err != nil {
		return fmt.Errorf("ошибка распаковки: %v", err)
	}
	
	if err := ioutil.WriteFile(tempFile, decompressedData, 0644); err != nil {
		return fmt.Errorf("ошибка записи временного файла: %v", err)
	}
	
	// Вычисляем хеш распакованного файла
	decompressedHash, err := CalculateFileHash(tempFile)
	if err != nil {
		return fmt.Errorf("ошибка вычисления хеша распакованного файла: %v", err)
	}
	
	// Сравниваем хеши
	if originalHash != decompressedHash {
		return fmt.Errorf("хеши не совпадают: исходный=%s, распакованный=%s", originalHash, decompressedHash)
	}
	
	fmt.Printf("Проверка целостности пройдена для %s\n", originalPath)
	return nil
}

func (ic *ImageCompressor) GetStatsJSON() []byte {
    stats := ic.GetStats()
    data, _ := json.Marshal(map[string]interface{}{
        "files_processed": stats.FilesProcessed,
        "compression_ratio": stats.CompressionRatio,
    })
    return data
}

func main() {
	var (
		inputPath   = flag.String("input", "", "Путь к входному файлу или директории")
		outputDir   = flag.String("output", "./compressed", "Директория для сжатых файлов")
		algorithm   = flag.String("algorithm", "gzip", "Алгоритм сжатия (gzip или zlib)")
		mode        = flag.String("mode", "compress", "Режим работы (compress, decompress, verify)")
		workers     = flag.Int("workers", 4, "Количество воркеров для параллельной обработки")
		batchFiles  = flag.String("batch", "", "Файл со списком путей для пакетной обработки (один путь на строку)")
	)
	flag.Parse()

	if *inputPath == "" {
		log.Fatal("Необходимо указать входной файл или директорию с помощью -input")
	}

	// Определяем алгоритм сжатия
	var algo CompressionAlgorithm
	switch strings.ToLower(*algorithm) {
	case "gzip":
		algo = GZIP
	case "zlib":
		algo = ZLIB
	default:
		log.Fatal("Поддерживаемые алгоритмы: gzip, zlib")
	}

	// Создаем компрессор
	compressor := NewImageCompressor(algo, *outputDir)

	// Выполняем операцию в зависимости от режима
	switch *mode {
	case "compress":
		if *batchFiles != "" {
			// Пакетная обработка из файла
			batchData, err := ioutil.ReadFile(*batchFiles)
			if err != nil {
				log.Fatalf("Ошибка чтения файла со списком: %v", err)
			}
			
			lines := strings.Split(string(batchData), "\n")
			var filePaths []string
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line != "" {
					filePaths = append(filePaths, line)
				}
			}
			
			if err := compressor.ProcessBatch(filePaths, *workers); err != nil {
				log.Fatalf("Ошибка пакетной обработки: %v", err)
			}
		} else {
			// Обработка одного файла или директории
			fileInfo, err := os.Stat(*inputPath)
			if err != nil {
				log.Fatalf("Ошибка получения информации о файле: %v", err)
			}

			if fileInfo.IsDir() {
				if err := compressor.ProcessDirectory(*inputPath); err != nil {
					log.Fatalf("Ошибка обработки директории: %v", err)
				}
			} else {
				if err := compressor.CompressFile(*inputPath); err != nil {
					log.Fatalf("Ошибка сжатия файла: %v", err)
				}
			}
		}
		
		compressor.PrintStats()

	case "decompress":
		if err := compressor.DecompressFile(*inputPath); err != nil {
			log.Fatalf("Ошибка распаковки: %v", err)
		}

	case "verify":
		// Для проверки нужны оба файла - исходный и сжатый
		compressedPath := *inputPath
		originalPath := strings.TrimSuffix(compressedPath, filepath.Ext(compressedPath))
		
		if err := compressor.VerifyCompression(originalPath, compressedPath); err != nil {
			log.Fatalf("Ошибка проверки: %v", err)
		}

	default:
		log.Fatal("Поддерживаемые режимы: compress, decompress, verify")
	}
}