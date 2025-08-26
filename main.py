import os
import json
import argparse
from tika import parser
from openai import OpenAI

def execute_prompt(prompt_text, model_name):
    client = OpenAI(api_key=os.environ.get("OPEN_AI"))
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def process_single_file(file_path, output_folder, model_name):
    # Prompt templates
    prompt_basic_info = """
    you have a cv and we need to extract each of the following from pdf file:
    basic info which is name, country, contact , email, education, latest position and company name he used to work in and return null if not found
    output json format:
    {
    "name": "string"
    "country": "string"
    "contact": "integers"
    "email": "characters",

    "education": 
        "degree": "string"
        "major": "string"
        "minor": "string"
        "university": "string"
       "duration": "integers"
      "latest_position": "string",
      "latest_company": "string",
     EXAMPLES:
    {
      "name": "Rawan Farouk",
      "country": "Egypt",
      "contact": "+201091221840",
      "email": "rawan@gamil.com",
      "education": {
        "degree": "B.Sc.",
        "major": "Computer Science",
        "minor": "intelligent systems",
        "university": "Alexandria University",
        "duration": "2020-2025"
      },
      "latest_position": "Mobile Developer",
      "latest_company": "pharos",

    }
    """

    prompt_languages = """
    we need to extract languages like arabic, english, german,...etc and extract if you find proficiency, language code, proficiency code and return null if not found
    OUTPUT JSON FORMAT:
    "proficiency": "string"
    "language code": "characters"
    "proficiency code": "characters"

    EXAMPLES:
    {
        "languages": [{
          "language": "Arabic",
          "language_code": "ar",
          "proficiency": "Native",
          "proficiency_code": "c2"
        }]
    }
    """

    prompt_specialties = """
    Extract a list of specialties or areas of expertise from the provided file, the output should be a JSON object with a single key "specialties" mapping to an array of strings and return null if not found

      OUTPUT JSON FORMAT:
    "specialties" : ["string"]

    EXAMPLES:
    {"specialties": [{
        "Software Engineering",
        "Backend Development",
        "Scalable Solutions",
        "Mentoring",
        "Technical Writing"}]}
        """

    prompt_skills = """
    From the file find all the skills mentioned and list them
    put the list under the heading "skills," with each skill as a separate item in the list and return null if not found
    OUTPUT JSON FORMAT:
    "skills" : ["string"]
    EXAMPLES:
    "skills": 
     "Python programming",
    " Data Analysis",
     "Project Management",
     "Communication"
    """

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False

    filename = os.path.basename(file_path)
    json_filename = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(output_folder, json_filename)
    
    current_cv = {"file_name": filename}
    
    try:
        # Extract text using Tika
        print(f"Converting {filename} to text...")
        parsed = parser.from_file(file_path)
        cv_text = parsed.get("content", "").strip()
        
        if not cv_text:
            print(f" No content extracted from: {filename}")
            return False

        # Basic Info
        print("Extracting basic info...")
        prompt = f"{prompt_basic_info}\n\nText:\n{cv_text}"
        result = execute_prompt(prompt, model_name)
        current_cv["basic_info"] = json.loads(result) if result else None

        # Languages
        print("Extracting languages...")
        prompt = f"{prompt_languages}\n\nText:\n{cv_text}"
        result = execute_prompt(prompt, model_name)
        current_cv["languages"] = json.loads(result).get("languages", []) if result else []

        # Specialties
        print("Extracting specialties...")
        prompt = f"{prompt_specialties}\n\nText:\n{cv_text}"
        result = execute_prompt(prompt, model_name)
        current_cv["specialties"] = json.loads(result).get("specialties", []) if result else []

        # Skills
        print("Extracting skills...")
        prompt = f"{prompt_skills}\n\nText:\n{cv_text}"
        result = execute_prompt(prompt, model_name)
        current_cv["skills"] = json.loads(result).get("skills", []) if result else []

        # Save JSON output
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(current_cv, f, indent=4, ensure_ascii=False)

        print(f"Successfully processed and saved: {json_filename}")
        return True

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        return False

def main():
    """Main function that acts as the entry point"""
    parser = argparse.ArgumentParser(description="Extract structured data from CV files")
    parser.add_argument("input_path", help="Path to a CV file or directory containing CV files")
    parser.add_argument("model_name", help="OpenAI model to use (e.g., gpt-4, gpt-4o, gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    # Clean and validate the input path
    input_path = os.path.normpath(args.input_path)
    
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return
    
    # Set up output folder
    output_folder = os.path.join(os.path.dirname(input_path), "structured_candidate_data")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Input: {input_path}")
    print(f"Model: {args.model_name}")
    print(f"Output: {output_folder}")
    
    try:
        if os.path.isdir(input_path):
            print("Processing directory...")
            print("Directory processing not implemented in this version.")
        else:
            success = process_single_file(input_path, output_folder, args.model_name)
            if success:
                print(f"\nProcessing complete! Results saved to: {output_folder}")
            else:
                print("\nFailed to process the file.")
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your OpenAI API key and network connection.")

if __name__ == "__main__":
    main()