import type { PalmReadingResult } from "../types/palm";

export async function analyzePalm(imageBase64: string): Promise<PalmReadingResult> {
  const url = `${import.meta.env.VITE_API_BASE_URL}/api/analyze`;

  try {
    // Convert base64 to Blob
    const res = await fetch(imageBase64);
    if (!res.ok) throw new Error("Failed to read image data");
    const blob = await res.blob();
    
    // Create FormData
    const formData = new FormData();
    // Default to palm_image.jpg, the backend supports .jpg, .jpeg, .png
    // The blob type will dictate how the backend parses it.
    formData.append('file', blob, 'palm_image.jpg');
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data?.error?.message || "Analysis failed");
    }
    
    return data.result;
  } catch (err: any) {
    if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
      throw new Error("Could not connect to the analysis server. Please check your connection.");
    }
    throw err;
  }
}
