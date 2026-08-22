import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { ArrowRight, User, Calendar, MapPin, ChevronDown } from "lucide-react";
import { usePalmReading } from "../context/PalmReadingContext";
import type { UserDetails } from "../types/palm";

// ── Step progress indicator ───────────────────────────────────────────────────
function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-2 mb-8">
      {Array.from({ length: total }, (_, i) => (
        <div key={i} className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full transition-all duration-400 ${
              i < current
                ? "bg-mystic-400"
                : i === current
                ? "bg-mystic-500 w-5"
                : "bg-border"
            }`}
          />
        </div>
      ))}
      <span className="text-text-muted text-xs ml-2">
        Step {current + 1} of {total}
      </span>
    </div>
  );
}

// ── Input field wrapper ───────────────────────────────────────────────────────
function FormField({
  label,
  icon: Icon,
  error,
  children,
}: {
  label: string;
  icon: React.ElementType;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label className="label-mystic flex items-center gap-2">
        <Icon className="w-3 h-3 text-mystic-500" />
        {label}
      </label>
      {children}
      {error && (
        <p className="text-red-400 text-xs mt-1">{error}</p>
      )}
    </div>
  );
}

// ── Main Details Page ─────────────────────────────────────────────────────────
export default function DetailsPage() {
  const navigate  = useNavigate();
  const { dispatch } = usePalmReading();

  const [form, setForm] = useState<UserDetails>({
    name:         "",
    dateOfBirth:  "",
    placeOfBirth: "",
    gender:       "",
  });

  const [errors, setErrors] = useState<Partial<Record<keyof UserDetails, string>>>({});

  const updateField = (field: keyof UserDetails, value: string) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    if (errors[field]) setErrors((prev) => ({ ...prev, [field]: undefined }));
  };

  const validate = (): boolean => {
    const newErrors: Partial<Record<keyof UserDetails, string>> = {};
    if (!form.name.trim())         newErrors.name         = "Please enter your name.";
    if (!form.dateOfBirth)         newErrors.dateOfBirth  = "Please select your date of birth.";
    if (!form.placeOfBirth.trim()) newErrors.placeOfBirth = "Please enter your place of birth.";
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (!validate()) return;
    dispatch({ type: "SET_USER", payload: form });
    navigate("/capture");
  };

  return (
    <div className="min-h-dvh bg-void bg-mystic-aura flex items-center justify-center px-4 py-16">
      <motion.div
        className="w-full max-w-md"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        {/* Header */}
        <div className="mb-8">
          <StepIndicator current={0} total={3} />
          <h1 className="font-serif text-4xl text-text-primary mb-2">Your Details</h1>
          <p className="text-text-muted text-sm">
            This helps us personalise your palm reading experience.
          </p>
        </div>

        {/* Form card */}
        <div className="glass-card p-8 space-y-6">
          {/* Name */}
          <FormField label="Full Name" icon={User} error={errors.name}>
            <input
              type="text"
              className="input-mystic"
              placeholder="Enter your full name"
              value={form.name}
              onChange={(e) => updateField("name", e.target.value)}
              autoComplete="name"
            />
          </FormField>

          {/* Date of Birth */}
          <FormField label="Date of Birth" icon={Calendar} error={errors.dateOfBirth}>
            <input
              type="date"
              className="input-mystic [color-scheme:dark]"
              value={form.dateOfBirth}
              onChange={(e) => updateField("dateOfBirth", e.target.value)}
              max={new Date().toISOString().split("T")[0]}
            />
          </FormField>

          {/* Place of Birth */}
          <FormField label="Place of Birth" icon={MapPin} error={errors.placeOfBirth}>
            <input
              type="text"
              className="input-mystic"
              placeholder="City or town"
              value={form.placeOfBirth}
              onChange={(e) => updateField("placeOfBirth", e.target.value)}
            />
          </FormField>

          {/* Gender — optional */}
          <FormField label="Gender (Optional)" icon={ChevronDown}>
            <select
              className="input-mystic cursor-pointer"
              value={form.gender}
              onChange={(e) => updateField("gender", e.target.value)}
            >
              <option value="">Prefer not to say</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="non-binary">Non-binary</option>
              <option value="prefer-not-to-say">Rather not specify</option>
            </select>
          </FormField>
        </div>

        {/* CTA */}
        <motion.button
          className="btn-primary w-full flex items-center justify-center gap-3 mt-6 group"
          onClick={handleSubmit}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <span>Continue</span>
          <ArrowRight className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-1" />
        </motion.button>

        <p className="text-center text-text-muted text-xs mt-4">
          Your details are used only to personalise your reading and are never stored or shared.
        </p>
      </motion.div>
    </div>
  );
}
