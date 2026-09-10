using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace VeggieLife.Backend.Services.AI
{
    // =========================================================================
    // DTO Classes matching Python FastAPI CV Service JSON Contract
    // =========================================================================

    public class BoundingBoxCoordinates
    {
        [JsonPropertyName("x_min")]
        public int XMin { get; set; }

        [JsonPropertyName("y_min")]
        public int YMin { get; set; }

        [JsonPropertyName("x_max")]
        public int XMax { get; set; }

        [JsonPropertyName("y_max")]
        public int YMax { get; set; }
    }

    public class BoundingBoxItem
    {
        [JsonPropertyName("label")]
        public string Label { get; set; } = string.Empty;

        [JsonPropertyName("label_vi")]
        public string LabelVi { get; set; } = string.Empty;

        [JsonPropertyName("category")]
        public string Category { get; set; } = string.Empty;

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        [JsonPropertyName("box")]
        public BoundingBoxCoordinates Box { get; set; } = new();
    }

    public class IngredientSummary
    {
        [JsonPropertyName("name_en")]
        public string NameEn { get; set; } = string.Empty;

        [JsonPropertyName("name_vi")]
        public string NameVi { get; set; } = string.Empty;

        [JsonPropertyName("category")]
        public string Category { get; set; } = string.Empty;

        [JsonPropertyName("count")]
        public int Count { get; set; }

        [JsonPropertyName("max_confidence")]
        public double MaxConfidence { get; set; }
    }

    public class DetectionResponse
    {
        [JsonPropertyName("success")]
        public bool Success { get; set; }

        [JsonPropertyName("execution_time_ms")]
        public double ExecutionTimeMs { get; set; }

        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = string.Empty;

        [JsonPropertyName("ingredient_count")]
        public int IngredientCount { get; set; }

        [JsonPropertyName("detected_ingredients")]
        public List<IngredientSummary> DetectedIngredients { get; set; } = new();

        [JsonPropertyName("bounding_boxes")]
        public List<BoundingBoxItem> BoundingBoxes { get; set; } = new();
    }

    // =========================================================================
    // Interface & Service Implementation for ASP.NET Core 8 Web API
    // =========================================================================

    public interface IVeggieLifeVisionClient
    {
        Task<DetectionResponse?> DetectFridgeIngredientsAsync(Stream imageStream, string fileName, double? confidence = null);
        Task<bool> CheckServiceHealthAsync();
    }

    public class VeggieLifeVisionClient : IVeggieLifeVisionClient
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<VeggieLifeVisionClient> _logger;

        public VeggieLifeVisionClient(HttpClient httpClient, ILogger<VeggieLifeVisionClient> logger)
        {
            _httpClient = httpClient;
            _logger = logger;
        }

        public async Task<bool> CheckServiceHealthAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("/health");
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Lỗi kết nối tới YOLO CV Microservice");
                return false;
            }
        }

        public async Task<DetectionResponse?> DetectFridgeIngredientsAsync(
            Stream imageStream,
            string fileName,
            double? confidence = null)
        {
            try
            {
                using var formData = new MultipartFormDataContent();
                
                var streamContent = new StreamContent(imageStream);
                streamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
                formData.Add(streamContent, "file", fileName);

                if (confidence.HasValue)
                {
                    formData.Add(new StringContent(confidence.Value.ToString("0.00")), "confidence");
                }

                _logger.LogInformation("Gửi ảnh {FileName} tới YOLO Microservice...", fileName);
                
                var response = await _httpClient.PostAsync("/api/v1/cv/detect", formData);
                
                if (!response.IsSuccessStatusCode)
                {
                    var err = await response.Content.ReadAsStringAsync();
                    _logger.LogError("YOLO Microservice trả lỗi: {StatusCode} - {Error}", response.StatusCode, err);
                    return null;
                }

                var responseJson = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<DetectionResponse>(responseJson, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Lỗi trong quá trình gọi YOLO Vision Service");
                throw;
            }
        }
    }
}
